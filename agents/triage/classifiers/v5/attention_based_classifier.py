from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from libraries.ml.device import get_device


class GatedAttentionPooling(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        attention_heads: int = 1,
        dropout: float | None = None,
    ):
        super().__init__()
        self.attention_v = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]

        self.attention_u = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]

        if dropout is not None:
            assert 0 < dropout < 1, "Dropout must be between 0 and 1"
            self.attention_u.append(nn.Dropout(dropout))
            self.attention_v.append(nn.Dropout(dropout))

        self.attention_v = nn.Sequential(*self.attention_v)
        self.attention_u = nn.Sequential(*self.attention_u)
        self.attention_w = nn.Linear(hidden_dim, attention_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)

        A = v.mul(u)
        A = self.attention_w(A)
        A = nn.functional.softmax(A, dim=0)

        return A, x


class AttentionBasedClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dims: tuple[int, int] = (256, 128),
        n_classes: int = 5,
        dropout: float | None = None,
        attention_heads: int = 1,
    ):
        super().__init__()

        attention_network = [nn.Linear(embedding_dim, hidden_dims[0]), nn.ReLU()]

        if dropout is not None:
            assert 0 < dropout < 1, "Dropout must be between 0 and 1"
            attention_network.append(nn.Dropout(dropout))

        gated_attention_pooling = GatedAttentionPooling(
            input_dim=hidden_dims[0],
            hidden_dim=hidden_dims[1],
            attention_heads=attention_heads,
            dropout=dropout,
        )
        attention_network.append(gated_attention_pooling)
        self.attention_network = nn.Sequential(*attention_network)
        self.classification_layer = nn.Linear(
            hidden_dims[0] * attention_heads, n_classes
        )

    def forward(self, x: torch.Tensor, return_attention_weights: bool = False) -> Any:
        A, h = self.attention_network(x)
        A = torch.transpose(A, 1, 0)
        z = torch.mm(A, h)
        z_flat = z.view(1, -1)
        logits = self.classification_layer(z_flat)
        if return_attention_weights:
            return logits, A, z
        return logits


class AttentionBasedClassifierWrapper(nn.Module):
    def __init__(
        self,
        attention_net: AttentionBasedClassifier,
        st_model_name: str = "all-MiniLM-L12-v2",
    ):
        super().__init__()
        device = get_device()
        self.attention_net = attention_net.to(device)
        self.sentence_transformer = SentenceTransformer(st_model_name, device=device)
        # Freeze SentenceTransformer so only the attention net is trained
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

    def forward(
        self, chat_history: list[str], return_attention_weights: bool = False
    ) -> torch.Tensor:
        x = self.sentence_transformer.encode(
            chat_history, convert_to_tensor=True, show_progress_bar=False
        )
        return self.attention_net(x, return_attention_weights)

    def predict_probab(
        self, chat_history: list[str], return_logits: bool = False
    ) -> Any:
        with torch.no_grad():
            outputs = self.forward(chat_history, return_attention_weights=return_logits)
            if return_logits:
                logits, attention_weights, attention_pooled_representation = outputs
            else:
                logits = outputs
            probs = nn.functional.softmax(logits, dim=1)

        if return_logits:
            return probs, logits, attention_weights, attention_pooled_representation
        return probs
