import logging
import os
import time
from datetime import datetime

import mlflow
import torch

from agents.triage.classifiers.v7.config import ClassifierV7Settings
from agents.triage.classifiers.v7.device_utils import (
    get_device_config,
    get_training_precision,
    move_to_mps,
)
from agents.triage.classifiers.v7.gemma3_seq_cls import (
    Gemma3TextForSequenceClassification,
)
from agents.triage.data_sets.loader import ID2LABEL, LABEL2ID

# Set tokenizers parallelism to avoid warnings during training
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)
import numpy as np
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def setup_mlflow_tracking(model_name: str) -> str:
    mlflow.set_experiment("triage_classifier")
    mlflow.dspy.autolog()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{model_name}_{timestamp}"


def log_training_parameters(sample_size: int, eval_sample_size: int, model_name: str, trainset_size: int, testset_size: int):
    mlflow.log_param("version", "v7")
    mlflow.log_param("model", model_name)
    mlflow.log_param("samples", sample_size)
    mlflow.log_param("eval_samples", eval_sample_size)
    mlflow.log_param("train_examples", trainset_size)
    mlflow.log_param("test_examples", testset_size)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)

    agent_names = ["BillingAgent", "ClaimsAgent", "PolicyAgent", "EscalationAgent", "ChattyAgent"]
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

    for i, agent_name in enumerate(agent_names):
        if i < len(precision_per_class):
            metrics[f'{agent_name.lower()}_precision'] = precision_per_class[i]
            metrics[f'{agent_name.lower()}_recall'] = recall_per_class[i]
            metrics[f'{agent_name.lower()}_f1'] = f1_per_class[i]
            metrics[f'{agent_name.lower()}_support'] = support[i]

    return metrics

def perform_fine_tuning(trainset: list, testset: list):
    v7_settings = ClassifierV7Settings()

    logger.info(f"Starting HuggingFace Gemma3 fine-tuning with {len(trainset)} examples")
    logger.info(f"Model: {v7_settings.get_model_name()}")
    logger.info(f"4-bit: {v7_settings.use_4bit}, QAT: {v7_settings.use_qat_model}")

    start_time = time.time()

    model_name = v7_settings.get_model_name()
    logger.info(f"Loading model and tokenizer: {model_name}")
    if v7_settings.use_qat_model:
        logger.info("Using QAT model variant")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if v7_settings.use_4bit and not v7_settings.use_qat_model and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    device_map, dtype = get_device_config()

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = v7_settings.n_classes
    config.label2id = LABEL2ID
    config.id2label = ID2LABEL

    model = Gemma3TextForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="eager"
    )

    # Needed for AutoModelForSequenceClassification.from_pretrained to find our class
    model.config.auto_map = {
        "AutoModel": "gemma3_seq_cls.Gemma3TextForSequenceClassification"
    }

    # The classifier head must be saved alongside LoRA adapters
    assert hasattr(model, "classifier")
    modules_to_save = ["classifier"]
    lora_config = LoraConfig(
        lora_alpha=v7_settings.lora_alpha,
        lora_dropout=v7_settings.lora_dropout,
        r=v7_settings.lora_r,
        bias="none",
        target_modules="all-linear",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)
    model = move_to_mps(model, device_map)
    model.train()
    mlflow.log_params(lora_config.to_dict())
    model.print_trainable_parameters()

    logger.info(f"Num train examples: {len(trainset)}")
    logger.info(f"Num eval examples: {len(testset)}")

    train_texts = [ex.chat_history for ex in trainset]
    train_labels = [LABEL2ID[ex.target_agent] for ex in trainset]

    eval_texts = [ex.chat_history for ex in testset]
    eval_labels = [LABEL2ID[ex.target_agent] for ex in testset]

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=v7_settings.max_length,
            return_tensors=None
        )
        model_inputs["labels"] = examples["labels"]
        return model_inputs

    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    eval_dataset = Dataset.from_dict({"text": eval_texts, "labels": eval_labels})

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=v7_settings.output_dir,
        num_train_epochs=v7_settings.num_epochs,
        per_device_train_batch_size=v7_settings.batch_size,
        per_device_eval_batch_size=v7_settings.batch_size,
        gradient_accumulation_steps=v7_settings.gradient_accumulation_steps,
        optim="adamw_torch",
        learning_rate=v7_settings.learning_rate,
        lr_scheduler_type="constant",
        **get_training_precision(),
        logging_steps=10,
        save_total_limit=1,
        eval_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        report_to="mlflow",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=v7_settings.max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving merged model (base + LoRA) to {v7_settings.output_dir}")
    best_model = trainer.model
    best_model = best_model.merge_and_unload()
    best_model.save_pretrained(v7_settings.output_dir)
    tokenizer.save_pretrained(v7_settings.output_dir)

    logger.info("Gemma3 fine-tuning completed.")

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")
    mlflow.log_metric("training_time_seconds", training_time)

    return best_model, tokenizer

def show_training_info():
    v7_settings = ClassifierV7Settings()

    print(f"Model: {v7_settings.get_model_name()}")
    print("Method: HuggingFace LoRA fine-tuning")
    print(f"LoRA rank: {v7_settings.lora_r}, Alpha: {v7_settings.lora_alpha}")
    print(f"Learning rate: {v7_settings.learning_rate}, Epochs: {v7_settings.num_epochs}")
    print(f"QAT enabled: {v7_settings.use_qat_model}")
    print(f"Output: {v7_settings.output_dir}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("GPU: Metal Performance Shaders (MPS) on Mac")
    else:
        print("Device: CPU (training will be slow)")
