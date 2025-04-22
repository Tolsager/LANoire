import mediapipe as mp
import numpy as np
from typing import List

from transformers import TimesformerModel
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import wandb

from LANoire import dataset, utils

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


def extract_face_mediapipe(frame):
    results = face_detector.process(frame)
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        return x1, y1, x2, y2
    return None


def get_bounding_boxes(frames: List[np.ndarray]):
    bboxes = []
    for frame in frames:
        result = extract_face_mediapipe(frame)
        if result is not None:
            x1, y1, x2, y2 = result
            result = frame[y1:y2, x1:x2]
        bboxes.append(result)

    return bboxes


class VideoDM(L.LightningDataModule):
    def __init__(self, bounding_boxes: np.ndarray, batch_size: int = 8):
        self.train_batch_size = batch_size
        self.eval_batch_size = int(1.5 * batch_size)
        self.bounding_boxes = bounding_boxes

    def setup(self) -> None:
        self.train_indices, self.val_indices, self.test_indices = (
            dataset.get_data_split_ids()
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.train_batch_size, **self.kwargs
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.eval_batch_size, **self.kwargs
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.eval_batch_size, **self.kwargs
        )


class VideoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        self.model.train()
        self.results = []

    def forward(self, model_input: list[np.ndarray]):
        """
        video_frames: list of numpy arrays (num_frames, height, width, channels)
        """
        output = self.model(model_input).last_hidden_state
        return output

    def test_step(self, batch: tuple):
        idx, frames = batch

        result = self(frames)
        self.results.append(result.mean(dim=1))

    def on_test_epoch_end(self):
        embeddings = torch.cat(self.results, dim=0).cpu
        utils.save_pickle("video_embeddings.pkl", embeddings)


class VideoMLP(L.LightningModule):
    def __init__(self, video_embeddings: torch.Tensor, hidden_size: int = 512):
        super().__init__()

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.embeds = torch.nn.Embedding.from_pretrained(video_embeddings)
        self.embed_dim = self.embeds.embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 1),
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
        self.log(
            "train_acc",
            accuracy(preds, targets, task="binary"),
            on_step=False,
            on_epoch=True,
        )
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
        return torch.optim.AdamW(self.mlp.parameters(), lr=5e-4)
