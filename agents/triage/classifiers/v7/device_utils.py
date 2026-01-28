import torch


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    return torch.backends.mps.is_available()


def no_grad():
    return torch.no_grad()


def get_device_config() -> tuple[str | None, torch.dtype]:
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        print(f"Using CUDA device with {dtype}")
    elif torch.backends.mps.is_available():
        # MPS doesn't support device_map="auto" and needs float32
        device_map = None
        dtype = torch.float32
        print("Using MPS device with float32")
    else:
        device_map = None
        dtype = torch.float32
        print("Using CPU device with float32")

    return device_map, dtype


def move_to_mps(model, device_map: str | None = None):
    # Only move to MPS if device_map wasn't used (CUDA uses device_map="auto")
    if torch.backends.mps.is_available() and device_map is None:
        model = model.to("mps")
    return model


def get_training_precision() -> dict:
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    # MPS doesn't handle fp16/bf16 training well
    if has_mps:
        return {
            "fp16": False,
            "bf16": False,
        }

    return {
        "fp16": has_cuda,
        "bf16": has_cuda and torch.cuda.get_device_capability()[0] >= 8,
    }
