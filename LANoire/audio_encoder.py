import torchaudio
from torchmetrics.classification import BinaryAccuracy
import transformers
import typing
from LANoire.dataset import LANoireDataset
from LANoire.utils import save_pickle
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
import wandb
from LANoire.dataset import get_data_split_ids

import lightning as L
import torch

class  WhisperAudioDS(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = "data/raw/data.json"):
        self.processor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        self.sr = 16_000
        self.ds = LANoireDataset(data_dir)
        self.resampler = torchaudio.transforms.Resample(orig_freq=44_100, new_freq=self.sr)
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, typing.Any]:
        audio = self.ds[idx][0]["answer"]
        audio = self.resampler(audio)

        features = self.processor(
            audio, sampling_rate=self.sr, return_tensors="pt"
        )
        return features.input_features[0]

class  Wav2Vec2DS(torch.utils.data.Dataset):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.sr = 16_000
        self.ds = LANoireDataset("data/raw/data.json")
        self.resampler = torchaudio.transforms.Resample(orig_freq=44_100, new_freq=self.sr)
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, typing.Any]:
        audio = self.ds[idx][0]["answer"]
        audio = self.resampler(audio)

        features = self.processor(
            audio, sampling_rate=self.sr, return_tensors="pt", padding="max_length", max_length=64_000, truncation=True
        )
        return features.input_values[0]

class Wav2Vec2DataModule(L.LightningDataModule):
    def __init__(
        self, batch_size: int = 16
    ):
        super().__init__()
        self.batch_size = batch_size

    def test_dataloader(self):
        ds = Wav2Vec2DS()
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False
        )

class Wav2Vec2Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.test_step_outputs = []
        self.model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.model(input_features).last_hidden_state.mean(dim=1)
    
    def test_step(self, batch: dict):
        embeds = self(batch)
        self.test_step_outputs.append(embeds)
    
    def on_test_epoch_end(self):
        embeddings = torch.cat(self.test_step_outputs, dim=0)
        save_pickle("data/processed/wav2vec2_embeddings.pkl", embeddings)

class WhisperDataModule(L.LightningDataModule):
    def __init__(
        self, batch_size: int = 16, num_workers: int = 7
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def test_dataloader(self):
        ds = WhisperAudioDS()
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
class WhisperModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.test_step_outputs = []
        self.model = AutoModel.from_pretrained("openai/whisper-tiny")

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(input_features).last_hidden_state.mean(dim=1)
    
    def test_step(self, batch: dict):
        embeds = self(batch)
        self.test_step_outputs.append(embeds)
    
    def on_test_epoch_end(self):
        embeddings = torch.cat(self.test_step_outputs, dim=0)
        save_pickle("data/processed/Whisper_embeddings.pkl", embeddings)

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

class ClapEeDs(torch.utils.data.Dataset):
    def __init__(self, model_name: str = "laion/clap-htsat-fused", json_path = "data/raw/data.json", data_dir = "data/raw"):
        self.processor = transformers.ClapFeatureExtractor.from_pretrained(model_name)
        self.CLAP_sr = 48000
        self.ds = LANoireDataset(json_path=json_path, data_dir=data_dir)
        self.resampler = torchaudio.transforms.Resample(orig_freq=44_100, new_freq=self.CLAP_sr)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, typing.Any]:
        audios, label = self.ds[idx]
        audio = audios["answer"]
        audio = self.resampler(audio)

        features = self.processor(
            audio, sampling_rate=self.CLAP_sr, return_tensors="pt"
        )
        features["input_features"] = features["input_features"][0]
        features["is_longer"] = features["is_longer"][0]
        return features, label

class ClapEeDm(L.LightningDataModule):
    def __init__(
        self, train_batch_size: int = 32, eval_batch_frac: float = 1.5, **kwargs
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = int(train_batch_size * eval_batch_frac)
        self.kwargs = kwargs

    def setup(self, stage: str = None) -> None:
        self.train_indices, self.val_indices, self.test_indices = get_data_split_ids("/work3/s204135/data/raw/data.json")
        ds = ClapEeDs(json_path="/work3/s204135/data/raw/data.json", data_dir="/work3/s204135/data/raw")
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

class CLAPModelEE(L.LightningModule):
    def __init__(self, model_name: str = "laion/clap-htsat-fused", lr: float = 2e-5):
        super().__init__()
        self.model = transformers.ClapAudioModelWithProjection.from_pretrained(
            model_name
        )
        self.fc1 = torch.nn.Linear(512, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.lr = lr
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_features: torch.Tensor, is_longer: torch.Tensor) -> torch.Tensor:
        embeds = self.model(input_features=input_features, is_longer=is_longer).audio_embeds
        return self.fc1(embeds).squeeze(dim=1)
        
    # def test_step(self, batch: dict):
    #     input_features = batch["input_features"]
    #     is_longer = batch["is_longer"]
    #     embeds = self(input_features, is_longer)
    #     self.test_step_outputs.append(embeds)
    
    # def on_test_epoch_end(self):
    #     embeddings = torch.cat(self.test_step_outputs, dim=0)
    #     save_pickle("data/processed/CLAP_embeddings.pkl", embeddings)

    def training_step(self, batch: tuple[dict, torch.Tensor]) -> torch.Tensor:
        features, y = batch
        y = y.float()
        pred = self(**features)
        loss = self.criterion(pred, y)
        self.train_acc(self.sigmoid(pred), y.int())
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: tuple[dict, torch.Tensor]) -> torch.Tensor:
        if self.global_step == 0:
            wandb.define_metric("val_acc", summary="max")
        features, y = batch
        y = y.float()
        pred = self(**features)
        self.val_acc(pred, y.int())
        loss = self.criterion(pred, y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


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
