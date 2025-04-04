from utils import save_pickle

from transformers import DistilBertModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L

class TextEncoder(L.LightningModule):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.embedding_outputs = []

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
        save_pickle("distilbert_embeds.pkl", embeddings)


class TextMLP(L.LightningModule):
    def __init__(self, embeds: torch.tensor, hidden_size: int = 768):
        super().__init__()

        self.embeds = nn.Embedding.from_pretrained(embeddings=embeds)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, idx: torch.tensor):
        embeds = self.embeds(idx)
        logits = self.mlp(embeds)
        return logits
    
    def training_step(self, batch, batch_idx: int):
        inputs, target = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target)
        return loss

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch
        output = self(inputs)
        loss = F.cross_entropy(output, targets)
        self.log("validation_loss", loss)

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch
        output = self(inputs)
        loss = F.cross_entropy(output, targets)
        self.log("test loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.mlp.parameters(), lr=0.005)
