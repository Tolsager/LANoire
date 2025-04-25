import lightning as L
import torch

from LANoire import dataset, video_encoder, utils
from torch.utils.data import Subset, DataLoader
from train_unimodal import get_model_arch

if __name__ == '__main__':
    batch_size: int = 16
    max_epochs: int = 100
    lr: float = 2e-5
    num_frames = 16

    train_ids, validation_ids, test_ids = dataset.get_data_split_ids(json_path="/work3/s204135/data/raw/data.json")
    ds = dataset.LANoireVideoDataset(json_path="/work3/s204135/data/raw/data.json", num_frames=num_frames, feature_extraction_level="bounding_box")
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    model = video_encoder.VideoModel(lr=lr, model_name="MCG-NJU/videomae-base-finetuned-kinetics")

    model_arch = get_model_arch(model, (1, num_frames, 3, 224, 224), dtypes=[torch.float])

    wandb_logger = utils.setup_logger(tags=["unimodal", "video", "end-to-end", "videomae"], config={"lr": lr, "batch_size": batch_size}, note=model_arch)

    trainer = L.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
