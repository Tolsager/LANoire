import numpy.typing as npt
import torchaudio
import transformers
import typing
from LANoire.dataset import LANoireDataset
from LANoire.utils import save_pickle

import lightning as L
import torch

def get_whisper_embeddings(audio: npt.NDArray, sampling_rate: int):
    model = transformers.WhisperFeautureExtractor().from_pretrained("openai/whisper-tiny.en")
    return model(audio, sampling_rate=sampling_rate)


class CLAPAudioDS(torch.utils.data.Dataset):
    def __init__(self, model_name: str = "laion/clap-htsat-fused"):
        self.processor = transformers.ClapFeatureExtractor.from_pretrained(model_name)
        self.CLAP_sr = 48000
        self.ds = LANoireDataset("data/raw/data.json")
        self.resampler = torchaudio.transforms.Resample(orig_freq=44_100, new_freq=self.CLAP_sr)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, typing.Any]:
        audio = self.ds[idx][0]["answer"]
        # audio = audio["answer"].numpy()
        # audio = librosa.resample(audio, orig_sr=22050, target_sr=self.CLAP_sr)
        audio = self.resampler(audio)

        features = self.processor(
            audio, sampling_rate=self.CLAP_sr, return_tensors="pt"
        )
        features["input_features"] = features["input_features"][0]
        features["is_longer"] = features["is_longer"][0]
        return features


class CLAPModel(L.LightningModule):
    def __init__(self, model_name: str = "laion/clap-htsat-fused"):
        super().__init__()
        self.model = transformers.ClapAudioModelWithProjection.from_pretrained(
            model_name
        )
        self.test_step_outputs = []

    def forward(self, input_features: torch.Tensor, is_longer: torch.Tensor) -> torch.Tensor:
        return self.model(input_features=input_features, is_longer=is_longer).audio_embeds
    
    def test_step(self, batch: dict):
        input_features = batch["input_features"]
        is_longer = batch["is_longer"]
        embeds = self(input_features, is_longer)
        self.test_step_outputs.append(embeds)
    
    def on_test_epoch_end(self):
        embeddings = torch.cat(self.test_step_outputs, dim=0)
        save_pickle("data/processed/CLAP_embeddings.pkl", embeddings)



class CLAPDataModule(L.LightningDataModule):
    def __init__(
        self, model_name: str = "laion/clap-htsat-fused", batch_size: int = 16, num_workers: int = 7, pin_memory: bool = False, persistent_workers: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def test_dataloader(self):
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
