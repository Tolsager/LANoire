import lightning as L
from LANoire.feature_fusion import CAF
from LANoire.gmu import BimodalGMU, TrimodalGMU
import torch
from LANoire.utils import load_pickle, save_pickle
from torchmetrics.classification import BinaryAccuracy
import wandb
import torch.nn.functional as F

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