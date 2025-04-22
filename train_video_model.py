import lightning as L
import torch

from LANoire import dataset, video_encoder, utils
from torch.utils.data import Subset, DataLoader
from train_unimodal import get_model_arch



if __name__ == '__main__':
    batch_size: int = 16

    train_ids, validation_ids, test_ids = dataset.get_data_split_ids()
    ds = dataset.LANoireIndexDataset()
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)


    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3, persistent_workers=True)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=3, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3)

    vid_embeddings = utils.load_pickle("video_embeddings.pkl")
    model = video_encoder.VideoMLP(video_embeddings=vid_embeddings, hidden_size=512)

    model_arch = get_model_arch(model, (1,), dtypes=[torch.long])

    wandb_logger = utils.setup_logger(tags=["unimodal", "video"], config={"lr": 5e-4, "batch_size": 16}, note=model_arch)

    trainer = L.Trainer(max_epochs=100, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
