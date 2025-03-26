import numpy.typing as npt
import transformers
import glob
import os
import typing

import librosa
import lightning as L
import torch
"data\1_buyers_beware\clovis_galletta\q1\clovis_galletta_lie_0.mp3"

def get_whisper_embeddings(audio: npt.NDArray, sampling_rate: int):
    model = transformers.WhisperFeautureExtractor().from_pretrained("openai/whisper-tiny.en")
    return model(audio, sampling_rate=sampling_rate)


class CLAPAudioDS(torch.utils.data.Dataset):
    def __init__(self, model_name: str):
        self.file_paths = glob.glob("data/*/*/*/*.mp3")
        self.file_names = [os.path.split(p)[1] for p in self.file_paths]
        self.processor = transformers.ClapFeatureExtractor.from_pretrained(model_name)
        self.CLAP_sr = 48000

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, typing.Any]:
        file_path = self.file_paths[idx]
        audio, _ = librosa.load(file_path, sr=self.CLAP_sr)
        features = self.processor(
            audio, sampling_rate=self.CLAP_sr, return_tensors="pt"
        )
        features["input_features"] = features["input_features"][0]
        features["is_longer"] = features["is_longer"][0]
        file_name = self.file_names[idx]
        return {**features, "file_name": file_name}


class CLAPModel(L.LightningModule):
    def __init__(self, model_name: str = "laion/clap-htsat-fused"):
        super().__init__()
        self.model = transformers.ClapAudioModelWithProjection.from_pretrained(
            model_name
        )

    def forward(self, batch: dict):
        input_features = batch["input_features"]
        is_longer = batch["is_longer"]
        file_names = batch["file_name"]
        return (
            self.model(input_features=input_features, is_longer=is_longer).audio_embeds,
            file_names,
        )


class CLAPDataModule(L.LightningDataModule):
    def __init__(
        self, model_name: str = "laion/clap-htsat-fused", batch_size: int = 16, num_workers: int = 7, pin_memory: bool = False, persistent_workers: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def predict_dataloader(self):
        ds = CLAPAudioDS(self.model_name)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers
        )

def dict_CLAP_out(batches: list[tuple[torch.Tensor, list[str]]]) -> tuple[torch.Tensor, list[str]]:
    out_dict = {}
    for b in batches:
        for i, fn in enumerate(b[1]):
            out_dict[fn] = b[0][i]
    return out_dict