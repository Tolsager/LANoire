from LANoire.utils import save_pickle

from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification
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
        self.embeds = nn.Embedding.from_pretrained(embeddings=embeds)
        self.embed_size = self.embeds.embedding_dim

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

class TransEE(L.LightningModule):
    def __init__(self, model_name: str = "roberta-base", lr=1e-4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.model.train()
        self.lr = lr
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, model_input: dict):
        output = self.model(**model_input)
        return output.logits.squeeze(dim=1)
    
    def training_step(self, batch):
        text, label = batch
        label = label.float()
        question = text["q"]
        answer = text["a"]
        texts = [f"question: {q} answer: {a}" for q, a in zip(question, answer)]
        
        model_input = self.tokenize(texts)

        out = self(model_input)
        preds = F.sigmoid(out)
        loss = self.criterion(out, label)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", accuracy(preds, label, task="binary"), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch):
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        text, label = batch
        label = label.float()
        question = text["q"]
        answer = text["a"]
        texts = [f"question: {q} answer: {a}" for q, a in zip(question, answer)]

        model_input = self.tokenize(texts)

        out = self(model_input)
        preds = F.sigmoid(out)
        loss = self.criterion(out, label)
        self.log("val_loss", loss)
        self.log("val_acc", accuracy(preds, label, task="binary"))

    def test_step(self, batch: tuple):
        text, label = batch
        label = label.float()
        question = text["q"]
        answer = text["a"]
        texts = [f"question: {q} answer: {a}" for q, a in zip(question, answer)]

        model_input = self.tokenize(texts)

        embeddings = self(model_input)

        self.embedding_outputs.append(embeddings[:, 0, :])

    def on_test_epoch_end(self):
        embeddings = torch.cat(self.embedding_outputs, dim=0)
        save_pickle(f"data/processed/{self.embed_name}", embeddings)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def tokenize(self, text_data):
        model_inputs = self.tokenizer(
                text_data, padding=True, truncation=True, return_tensors="pt"
                )
        model_inputs["input_ids"] = model_inputs["input_ids"].to(self.model.device)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].to(self.model.device)
        return model_inputs
