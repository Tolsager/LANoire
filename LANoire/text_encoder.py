from utils import save_pickle

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
import wandb

from torchmetrics.functional import accuracy

class TextEncoder(L.LightningModule):
    def __init__(self, model_name: str = "distilbert-base-uncased", embed_name: str = "distilbert_embeds.pkl"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_outputs = []
        self.embed_name = embed_name

    def forward(self, model_input: dict):
        output = self.model(**model_input)
        return output.last_hidden_state

    def test_step(self, batch: tuple):
        text, label = batch
        question = text["q"]
        answer = text["a"]
        texts = [f"question: {q} answer: {a}" for q, a in zip(question, answer)]
        model_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        embeddings = self(model_input)

        self.embedding_outputs.append(embeddings[:, 0, :])

    def on_test_epoch_end(self):
        embeddings = torch.cat(self.embedding_outputs, dim=0)
        save_pickle(f"data/processed/{self.embed_name}", embeddings)


class TextMLP(L.LightningModule):
    def __init__(self, embeds: torch.tensor, hidden_size: int = 768):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss()
        self.embed_size = embeds.shape[-1]
        self.embeds = nn.Embedding.from_pretrained(embeddings=embeds)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.25),
            # nn.Linear(hidden_size, hidden_size),
            # nn.GELU(),
            # nn.Dropout(p=0.15),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, idx: torch.tensor):
        embeds = self.embeds(idx)
        logits = self.mlp(embeds)
        return logits
    
    def training_step(self, batch):
        inputs, targets = batch
        output = self(inputs).squeeze(1)
        loss = self.criterion(output, targets)
        preds = F.sigmoid(output)
        self.log("train_acc", accuracy(preds, targets, task="binary"), on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        inputs, targets = batch
        output = self(inputs).squeeze(1)
        loss = self.criterion(output, targets)
        preds = F.sigmoid(output)
        self.log("val_loss", loss)
        self.log("val_acc", accuracy(preds, targets, task="binary"))

    def test_step(self, batch):
        inputs, targets = batch
        output = self(inputs).squeeze(1)
        loss = self.criterion(output, targets)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.mlp.parameters(), lr=1e-4)
