import numpy.typing as npt
import transformers
import glob
import os
import typing

import librosa
import lightning as L
import torch

def get_whisper_embeddings(audio: npt.NDArray, sampling_rate: int):
    model = transformers.WhisperFeautureExtractor().from_pretrained("openai/whisper-tiny.en")
    return model(audio, sampling_rate=sampling_rate)



class CLAPAudioDS(torch.utils.data.Dataset):
    def __init__(self, audio_dir: str, model_name: str):
        self.audio_dir = audio_dir
        self.file_paths = glob.glob(f"{audio_dir}/*.wav")
        self.file_names = [os.path.split(p)[1] for p in self.file_paths]
        self.processor = transformers.ClapFeatureExtractor.from_pretrained(model_name)
        self.CLAP_sr = 48000

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, typing.Any]:
        file_path = self.file_paths[idx]
        audio, _ = librosa.load(file_path, sr=self.CLAP_sr)
        features = self.processor(
            audio, return_tensors="pt", sampling_rate=self.CLAP_sr
        )
        features["input_features"] = features["input_features"][0]
        features["is_longer"] = features["is_longer"][0]
        file_name = self.file_names[idx]
        return {**features, "file_name": file_name}


class CLAPModel(L.LightningModule):
    def __init__(self, model_name: str):
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
        self, model_name: str, audio_dir: str, batch_size: int, num_workers: int = 0, pin_memory: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def predict_dataloader(self):
        ds = CLAPAudioDS(self.audio_dir, self.model_name)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
        )