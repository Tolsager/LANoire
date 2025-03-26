import lightning as L
import numpy.typing as npt
import numpy as np
from utils import load_pickle
import torch
from sklearn.model_selection import train_test_split

class CLAPMLP(L.LightningModule):
    def __init__(self):
        super().__init__()


class CLAPMLPDM(L.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 7, pin_memory: bool = False, persistent_workers: bool = True):
        super().__init__()
    
    def prepare_data(self):
        self.ds_inputs = embed_dict_to_ds_inputs(load_pickle("CLAP_embeds.pkl"))

    def setup(self, stage: str = "fit"):
        labels = self.ds_inputs[1].numpy()
        indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(indices, stratify=labels, test_size=0.2)
        val_indices, test_indices = train_test_split(val_indices, stratify=labels[val_indices], test_size=0.5)

        
        if stage == "fit":
            embeds = self.ds_inputs[train_indices]
            labels = self.ds_inputs[1][train_indices]
            filenames = self.ds_inputs[2][train_indices]
            self.train_ds = CLAPMLPDS((embeds, labels, filenames), stage="fit")
        elif stage == "validate":
            embeds = self.ds_inputs[val_indices]
            labels = self.ds_inputs[1][val_indices]
            filenames = self.ds_inputs[2][val_indices]
            self.val_ds = CLAPMLPDS((embeds, labels, filenames), stage="fit")
        elif stage == "test":
            embeds = self.ds_inputs[test_indices]
            labels = self.ds_inputs[1][test_indices]
            filenames = self.ds_inputs[2][test_indices]
            self.test_ds = CLAPMLPDS((embeds, labels, filenames), stage="fit")
        
    
def embed_dict_to_ds_inputs(embed_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, npt.NDArray]:
    filenames = []
    embeddings = []
    for k, v in embed_dict.items():
        filenames.append(k)
        embeddings.append(v)
    labels = torch.tensor([0 if "_lie_" in f else 1 for f in filenames])
    embeddings = torch.stack(embeddings, dim=0)
    return (embeddings, labels, np.array(filenames))


class CLAPMLPDS(torch.utils.data.Dataset):
    def __init__(self, ds_inputs: tuple[torch.tensor, torch.tensor, npt.NDArray], stage: str = "fit"):
        super().__init__()
        self.ds_inputs = ds_inputs
        self.stage = stage
        if stage not in ["fit", "validate", "test"]:
            raise ValueError("stage must be one of 'fit', 'validate', 'test'")
    
    def __len__(self):
        return len(self.ds_inputs[1])
    
    def __getitem__(self, index: int):
        if self.stage == "fit":
            return self.ds_inputs[0][index], self.ds_inputs[1][index]
        elif self.state == "validate" or self.state == "test":
            return self.ds_inputs[0][index], self.ds_inputs[1][index], self.ds_inputs[2][index]
    



