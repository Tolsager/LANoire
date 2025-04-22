import lightning as L
from LANoire.utils import save_pickle
import wandb
from LANoire.dataset import LANoireIndexDataset, get_data_split_ids
import numpy.typing as npt
import numpy as np
from utils import load_pickle
import torch
from torchmetrics.classification import BinaryAccuracy

class Wav2vec2Mlp(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1):
        super().__init__()
        self.embeddings = load_pickle("data/processed/wav2vec2_embeddings.pkl")
        self.embeddings = torch.nn.Embedding.from_pretrained(
            self.embeddings, freeze=True
        )
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(768, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
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



class WhisperMlp(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1):
        super().__init__()
        self.embeddings = load_pickle("data/processed/Whisper_embeddings.pkl")
        self.embeddings = torch.nn.Embedding.from_pretrained(
            self.embeddings, freeze=True
        )
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(384, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)
        self.val_wrong = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
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
            save_pickle("data/processed/whisper_errors.pkl", self.val_wrong)


class ClapMlp(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.1):
        super().__init__()
        self.embeddings = load_pickle("data/processed/CLAP_embeddings.pkl")
        self.embeddings = torch.nn.Embedding.from_pretrained(
            self.embeddings, freeze=True
        )
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class EmbeddingDm(L.LightningDataModule):
    def __init__(
        self, train_batch_size: int = 32, eval_batch_frac: float = 1.5, **kwargs
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = int(train_batch_size * eval_batch_frac)
        self.kwargs = kwargs

    def setup(self, stage: str = None) -> None:
        self.train_indices, self.val_indices, self.test_indices = get_data_split_ids()
        ds = LANoireIndexDataset()
        self.train_ds = torch.utils.data.Subset(ds, self.train_indices)
        self.val_ds = torch.utils.data.Subset(ds, self.val_indices)
        self.test_ds = torch.utils.data.Subset(ds, self.test_indices)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.train_batch_size, shuffle=True, **self.kwargs
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.eval_batch_size, **self.kwargs
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.eval_batch_size, **self.kwargs
        )


def embed_dict_to_ds_inputs(
    embed_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, npt.NDArray]:
    filenames = []
    embeddings = []
    for k, v in embed_dict.items():
        filenames.append(k)
        embeddings.append(v)
    labels = torch.tensor([0 if "_lie_" in f else 1 for f in filenames])
    embeddings = torch.stack(embeddings, dim=0)
    return (embeddings, labels, np.array(filenames))
