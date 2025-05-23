import lightning as L
from LANoire.feature_fusion import CAF
from LANoire.gmu import BimodalGMU, TrimodalGMU
import torch
from LANoire.utils import load_pickle, save_pickle
from torchmetrics.classification import BinaryAccuracy
import wandb
import torch.nn.functional as F
from LANoire.feature_fusion import ContrastiveLossELI5

from transformers import AutoModel, ClapAudioModelWithProjection
from torchmetrics.functional import accuracy

class TextAudioGmu(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        super().__init__()
        text_embeddings_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.text_embeddings = torch.nn.Embedding.from_pretrained(
        text_embeddings_t, freeze=True
        )
        audio_embeddings_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.audio_embeddings = torch.nn.Embedding.from_pretrained(
        audio_embeddings_t, freeze=True
        )
        self.gmu = BimodalGMU(768, 512)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.fc = torch.nn.Linear(200, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, id: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_embeddings(id)
        audio_emb = self.audio_embeddings(id)
        x = self.gmu(text_emb, audio_emb)
        x = self.dropout(x)
        x = self.fc(x)
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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.01, batch_size: int = 50, temperature: float = 0.5, w_contr: float = 0.1):
        super().__init__()
        self.w_contr = w_contr
        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.text_projector = torch.nn.Linear(768, 512)
        self.CAF = CAF()
        self.fc1 = torch.nn.Linear(512, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.val_wrong = []
        self.contr_loss = ContrastiveLossELI5(batch_size, temperature)
        self.text_model = AutoModel.from_pretrained("roberta-base")
        self.audio_model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")

    def forward(self, x_audio, x_text, train: bool=False) -> torch.Tensor:
        audio_features = self.audio_model(**x_audio).audio_embeds
        text_features = self.text_model(**x_text).last_hidden_state[:, 0, :]
        text_features = self.text_projector(text_features)
        x = self.CAF(audio_features, text_features)
        x = self.fc1(x)
        if train:
            return x.squeeze(dim=1), audio_features, text_features
        return x.squeeze(dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.text_model.train()
        self.audio_model.train()
        x_audio, x_text, y = batch
        y = y.float()
        pred, audio_emb, text_emb = self(x_audio, x_text, train=True)
        contr_loss = self.contr_loss(audio_emb, text_emb)
        clf_loss = self.criterion(pred, y)
        loss = clf_loss + self.w_contr * contr_loss
        self.train_acc(self.sigmoid(pred), y.int())
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.text_model.eval()
        self.audio_model.eval()
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        x_audio, x_text, y = batch
        y = y.float()
        pred= self(x_audio, x_text, train=False)
        self.val_acc(pred, y.int())
        self.log("val_acc", self.val_acc)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_audio, x_text, y = batch
        y = y.float()
        pred = self(x_audio, x_text, train=False)
        self.test_acc(pred, y.int())
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    # def on_validation_epoch_end(self):
    #     if self.trainer.current_epoch == self.trainer.max_epochs - 1:
    #         save_pickle("data/processed/wav2vec2_errors.pkl", self.val_wrong)
    
class TextAudioCATee(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.01):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(768+512, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.val_wrong = []
        self.text_model = AutoModel.from_pretrained("roberta-base")
        self.audio_model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")

    def forward(self, x_audio, x_text) -> torch.Tensor:
        audio_features = self.audio_model(**x_audio).audio_embeds
        text_features = self.text_model(**x_text).last_hidden_state[:, 0, :]
        x = torch.cat([audio_features, text_features], dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x.squeeze(dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.text_model.train()
        self.audio_model.train()
        x_audio, x_text, y = batch
        y = y.float()
        pred = self(x_audio, x_text)
        loss = self.criterion(pred, y)
        self.train_acc(self.sigmoid(pred), y.int())
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.text_model.eval()
        self.audio_model.eval()
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        x_audio, x_text, y = batch
        y = y.float()
        pred= self(x_audio, x_text)
        self.val_acc(pred, y.int())
        self.log("val_acc", self.val_acc)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_audio, x_text, y = batch
        y = y.float()
        pred = self(x_audio, x_text)
        self.test_acc(pred, y.int())
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class TextAudioGMUee(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.01):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(200, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.gmu = BimodalGMU(768, 512)
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.val_wrong = []
        self.text_model = AutoModel.from_pretrained("roberta-base")
        self.audio_model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")

    def forward(self, x_audio, x_text) -> torch.Tensor:
        audio_features = self.audio_model(**x_audio).audio_embeds
        text_features = self.text_model(**x_text).last_hidden_state[:, 0, :]
        x = self.gmu(text_features, audio_features)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.text_model.train()
        self.audio_model.train()
        x_audio, x_text, y = batch
        y = y.float()
        pred = self(x_audio, x_text)
        loss = self.criterion(pred, y)
        self.train_acc(self.sigmoid(pred), y.int())
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.text_model.eval()
        self.audio_model.eval()
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        x_audio, x_text, y = batch
        y = y.float()
        pred= self(x_audio, x_text)
        self.val_acc(pred, y.int())
        self.log("val_acc", self.val_acc)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_audio, x_text, y = batch
        y = y.float()
        pred = self(x_audio, x_text)
        self.test_acc(pred, y.int())
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class TextVideoGmu(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        super().__init__()
        emb1_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        self.gmu = BimodalGMU(emb1_dim, emb2_dim)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.fc = torch.nn.Linear(200, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        x = self.gmu(emb1, emb2)
        x = self.dropout(x)
        x = self.fc(x)
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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class TextVideoCat(TextVideoGmu):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        L.LightningModule.__init__(self)
        emb1_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.fc = torch.nn.Linear(emb1_dim + emb2_dim, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        x = torch.cat([emb1, emb2], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

class TextVideoCaf(TextVideoGmu):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        L.LightningModule.__init__(self)
        emb1_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.caf = CAF()
        self.fc1 = torch.nn.Linear(emb1_dim, 512)
        self.fc2 = torch.nn.Linear(emb2_dim, 512)
        self.fc = torch.nn.Linear(512, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        x1 = self.fc1(emb1)
        x2 = self.fc2(emb2)
        x = self.caf(x1, x2)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

class TextAudioVideo(L.LightningModule):
    def __init__(
            self,
            text_model: str = "roberta-base",
            audio_model: str = "laion/clap-htsat-fused",
            video_model: str = "MCG-NJU/videomae-base-finetuned-kinetics",
            lr: float = 1e-4,
            feature_fusion: str = "CAF",
            batch_size: int = 32,
            temperature: float = 0.5
        ):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(text_model)
        self.audio_model = ClapAudioModelWithProjection.from_pretrained(audio_model)
        self.video_model = AutoModel.from_pretrained(video_model)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.dropout = torch.nn.Dropout(p=0.25)

        self.feature_fusion = feature_fusion

        match self.feature_fusion:
            case "CAF":
                self.text_projection = torch.nn.Linear(768, 512)
                self.audio_projection = torch.nn.Linear(512, 512)
                self.video_projection = torch.nn.Linear(768, 512)
                self.CAF = CAF(n_features=3, attention_dim=64)
                self.projector = torch.nn.Linear(512, 1)
                self.contr_loss = ContrastiveLossELI5(batch_size, temperature)
            case "GMU":
                self.GMU = TrimodalGMU(768, 512, 768, hidden_size=256)
                self.projector = torch.nn.Linear(256, 1)
            case "CONCAT":
                self.projector = torch.nn.Linear(768+512+768, 1)

    def forward(self, x_text, x_audio, x_video, train_caf: bool = False):
        text_features = self.text_model(**x_text).last_hidden_state[:, 0, :]
        audio_features = self.audio_model(**x_audio).audio_embeds
        video_features = self.video_model(x_video).last_hidden_state[:, 0, :]
        if self.feature_fusion == "CAF":
            text_features = self.text_projection(text_features)
            audio_features = self.audio_projection(audio_features)
            video_features = self.video_projection(video_features)
        fused = self.fuse_features(text_features, audio_features, video_features).squeeze(1)
        if train_caf:
            return fused, audio_features, text_features, video_features
        return fused

    def on_train_epoch_start(self):
        self.text_model.train()
        self.video_model.train()
        self.audio_model.train()
    
    def on_validation_epoch_start(self):
        self.text_model.eval()
        self.video_model.eval()
        self.audio_model.eval()


    def training_step(self, batch):
        if self.feature_fusion == "CAF":
            (out, audio_feat, text_feat, video_feat), label = self._shared_step(batch, caf_train=True)
        else:
            out, label = self._shared_step(batch)

        audio_text_loss = self.contr_loss(audio_feat, text_feat)
        audio_video_loss = self.contr_loss(audio_feat, video_feat)
        text_video_loss = self.contr_loss(text_feat, video_feat)

        loss = self.criterion(out, label) + 0.1 * 1/3 * (audio_text_loss + audio_video_loss + text_video_loss)
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
        self.log("test_acc", accuracy(pred, label, task="binary"))

    def _shared_step(self, batch, caf_train: bool = False):
        audio, text, video, label = batch
        return self(text, audio, video, caf_train), label

    def fuse_features(self, text_feat, audio_feat, video_feat):
        match self.feature_fusion:
            case "CAF":
                caf_features = self.CAF(text_feat, audio_feat, video_feat)
                return self.projector(self.dropout(caf_features))
            case "GMU":
                gmu_features = self.GMU(text_feat, audio_feat, video_feat)
                return self.projector(self.dropout(gmu_features))
            case "CONCAT":
                concat_features = torch.cat([text_feat, audio_feat, video_feat], dim=1)
                return self.projector(self.dropout(concat_features))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class AudioVideoGmu(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        super().__init__()
        emb1_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        self.gmu = BimodalGMU(emb1_dim, emb2_dim)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.fc = torch.nn.Linear(200, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        x = self.gmu(emb1, emb2)
        x = self.dropout(x)
        x = self.fc(x)
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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class AudioVideoCat(AudioVideoGmu):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        L.LightningModule.__init__(self)
        emb1_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.fc = torch.nn.Linear(emb1_dim + emb2_dim, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        x = torch.cat([emb1, emb2], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

class AudioVideoCaf(AudioVideoGmu):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        L.LightningModule.__init__(self)

        emb1_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.caf = CAF()
        self.fc1 = torch.nn.Linear(emb1_dim, 512)
        self.fc2 = torch.nn.Linear(emb2_dim, 512)
        self.fc = torch.nn.Linear(512, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        x1 = self.fc1(emb1)
        x2 = self.fc2(emb2)
        x = self.caf(x1, x2)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

class AllCat(AudioVideoGmu):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        L.LightningModule.__init__(self)
        emb1_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb3_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.emb3 = torch.nn.Embedding.from_pretrained(
        emb3_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        emb3_dim = self.emb2.embedding_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.fc = torch.nn.Linear(emb1_dim + emb2_dim + emb3_dim, 1)
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        emb3 = self.emb3(id)
        x = torch.cat([emb1, emb2, emb3], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

class AllCaf(AudioVideoGmu):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        L.LightningModule.__init__(self)
        emb1_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb3_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.emb3 = torch.nn.Embedding.from_pretrained(
        emb3_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        emb3_dim = self.emb2.embedding_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()
        self.caf = CAF(n_features=3)
        self.fc1 = torch.nn.Linear(emb1_dim, 512)
        self.fc2 = torch.nn.Linear(emb2_dim, 512)
        self.fc3 = torch.nn.Linear(emb3_dim, 512)
        self.fc = torch.nn.Linear(512, 1)
    
    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        emb3 = self.emb3(id)
        z1 = self.fc1(emb1)
        z2 = self.fc2(emb2)
        z3 = self.fc3(emb3)
        x = self.caf(z1, z2, z3)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)

class AllGmu(AudioVideoGmu):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1, weight_decay: float = 0.2):
        L.LightningModule.__init__(self)
        emb1_t = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.emb1 = torch.nn.Embedding.from_pretrained(
        emb1_t, freeze=True
        )
        emb2_t = load_pickle("video_embeddings_videomae.pkl")
        self.emb2 = torch.nn.Embedding.from_pretrained(
        emb2_t, freeze=True
        )
        emb3_t = load_pickle("data/processed/roberta_base_embeds.pkl")
        self.emb3 = torch.nn.Embedding.from_pretrained(
        emb3_t, freeze=True
        )
        emb1_dim = self.emb1.embedding_dim
        emb2_dim = self.emb2.embedding_dim
        emb3_dim = self.emb2.embedding_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []
        self.weight_decay = weight_decay
        self.sigmoid = torch.nn.Sigmoid()
        self.gmu = TrimodalGMU(emb1_dim, emb2_dim, emb3_dim)
        self.fc = torch.nn.Linear(200, 1)
    
    def forward(self, id: torch.Tensor) -> torch.Tensor:
        emb1 = self.emb1(id)
        emb2 = self.emb2(id)
        emb3 = self.emb3(id)
        x = self.gmu(emb1, emb2, emb3)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)
