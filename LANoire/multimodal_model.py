import lightning as L
from LANoire.feature_fusion import CAF
import torch
from LANoire.utils import load_pickle, save_pickle
from torchmetrics.classification import BinaryAccuracy
import wandb
import torch.nn.functional as F

from transformers import AutoModel, AutoFeatureExtractor
from torchmetrics.functional import accuracy

class TextAudioCat(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1):
        super().__init__()
        text_embeddings_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.text_embeddings = torch.nn.Embedding.from_pretrained(
        text_embeddings_t, freeze=True
        )
        audio_embeddings_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.audio_embeddings = torch.nn.Embedding.from_pretrained(
        audio_embeddings_t, freeze=True
        )
        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(512+768, 400)
        self.fc2 = torch.nn.Linear(400, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []

    def forward(self, id: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_embeddings(id)
        audio_emb = self.audio_embeddings(id)
        x = torch.cat([text_emb, audio_emb], dim=1)
        x = self.dropout(x)
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y = y.float()
        pred = self(x)
        loss = self.criterion(pred, y)
        self.train_acc(self.sigmoid(pred), y.int())
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        x, y = batch
        y = y.float()
        pred = self(x)
        self.val_acc(pred, y.int())
        loss = self.criterion(pred, y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss)

        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            pred = self.sigmoid(pred) > 0.5
            self.val_wrong.extend(x[pred != y].tolist())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_validation_epoch_end(self):
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            save_pickle("data/processed/wav2vec2_errors.pkl", self.val_wrong)

class TextAudioCAF(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.01):
        super().__init__()
        text_embeddings_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.text_embeddings = torch.nn.Embedding.from_pretrained(
        text_embeddings_t, freeze=True
        )
        audio_embeddings_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.audio_embeddings = torch.nn.Embedding.from_pretrained(
        audio_embeddings_t, freeze=True
        )
        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.text_projector = torch.nn.Linear(768, 512)
        self.CAF = CAF()
        self.fc1 = torch.nn.Linear(512, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.val_wrong = []

    def forward(self, id: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_embeddings(id)
        text_emb = self.text_projector(text_emb)
        audio_emb = self.audio_embeddings(id)
        x = self.CAF(text_emb, audio_emb)
        x = self.fc1(x)
        return x.squeeze(dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y = y.float()
        pred = self(x)
        loss = self.criterion(pred, y)
        self.train_acc(self.sigmoid(pred), y.int())
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        x, y = batch
        y = y.float()
        pred = self(x)
        self.val_acc(pred, y.int())
        loss = self.criterion(pred, y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss)

        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            pred = self.sigmoid(pred) > 0.5
            self.val_wrong.extend(x[pred != y].tolist())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_validation_epoch_end(self):
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            save_pickle("data/processed/wav2vec2_errors.pkl", self.val_wrong)


class TextAudioVideo(L.LightningModule):
    def __init__(
            self,
            text_model: str = "roberta-base",
            audio_model: str = "laion/clap-htsat-fused",
            video_model: str = "MCG-NJU/videomae-base-finetuned-kinetics",
            lr: float = 1e-4,
            feature_fusion: str = "CAF"
        ):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(text_model)
        self.audio_model = AutoFeatureExtractor.from_pretrained(audio_model)
        self.video_model = AutoModel.from_pretrained(video_model)
        self.lr = lr

        self.feature_fusion = feature_fusion

        match self.feature_fusion:
            case "CAF":
                self.text_projection = torch.nn.Linear(768, 512)
                self.audio_projection = torch.nn.Linear(512, 512)
                self.video_projection = torch.nn.Linear(768, 512)
                self.CAF = CAF(n_features=3, attention_dim=64)
            case "GMU":
                self.GMU = ...
            case "CONCAT":
                self.projector = torch.nn.Linear(768+512+768, 1)

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x_text, x_audio, x_video):
        text_features = self.text_model(**x_text)
        audio_features = self.audio_model(x_audio)
        video_features = self.video_model(**x_video)
        fused_features = self.feature_fusion(text_features, audio_features, video_features)
        return self.classifier(fused_features)

    def training_step(self, batch):
        
        out, label = self._shared_step(batch)
        loss = self.criterion(out, label)
        pred = F.sigmoid(out)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", accuracy(pred, label, task="binary"), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch):
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        out, label = self._shared_step(batch)
        loss = self.criterion(out, label)
        pred = F.sigmoid(out)
        self.log("val_loss", loss)
        self.log("val_acc", accuracy(pred, label, task="binary"))

    def test_step(self, batch):

        out, label = self._shared_step(batch)
        pred = F.sigmoid(out)


    def _shared_step(self, batch):
        audio, text, video, label = batch
        return self(text, audio, video), label


    def feature_fusion(self, text_feat, audio_feat, video_feat):
        match self.feature_fusion:
            case "CAF":
                caf_text = self.text_projection(text_feat)
                caf_audio = self.audio_projection(audio_feat)
                caf_video = self.video_projection(video_feat)
                fused_features = self.CAF(caf_text, caf_audio, caf_video)
                return fused_features
            case "GMU":
                ...
            case "CONCAT":
                concat_features = torch.cat([text_feat, audio_feat, video_feat], dim=1)
                return self.projector(concat_features)

        
        return
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    