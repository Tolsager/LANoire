import sys
sys.path.append("")
from LANoire.dataset import LANoireDataset, Modality
from utils import save_pickle

from transformers import DistilBertModel, AutoTokenizer
import torch.nn as nn
import torch
import lightning as L


class TextEncoder(L.LightningModule):
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_size: int = 768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        # self.fc = nn.Linear(hidden_size, 2)
        self.embedding_outputs = []


    def forward(self, batch: tuple[dict|int]):
        text, label = batch
        question = text["q"]
        answer = text["a"]
        texts = f"question: {question} answer: {answer}"

        input_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        output = self.model(**input_ids)

        return output.last_hidden_state, label
    
    def test_step(self, batch: dict):
        embedding, label = self(batch)
        self.embedding_outputs.append((embedding[:, 0, :], label))

    def on_test_epoch_end(self, batch: dict):
        embeddings = torch.cat([x[0] for x in self.embedding_outputs], dim=0)
        labels = [x[1] for x in self.embedding_outputs]
        save_pickle("distilbert-embeds.pkl", (embeddings, labels))


if __name__ == '__main__':
    ds = LANoireDataset(modalities=(Modality.TEXT,))

    text_encoder = TextEncoder()

    for text, label in ds:
        text_encoder((text, label))
