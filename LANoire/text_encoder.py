from transformers import DistilBertModel, AutoTokenizer
from LANoire.dataset import LANoireDataset, Modality


import lightning as L


class TextEncoder(L.LightningModule):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DistilBertModel.from_pretrained(model_name)

    def forward(self, batch: dict):
        return


ds = LANoireDataset(modalities=(Modality.TEXT,))
