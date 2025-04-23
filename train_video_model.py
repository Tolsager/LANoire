import lightning as L
import torch

from LANoire import dataset, video_encoder, utils
from torch.utils.data import Subset, DataLoader
from train_unimodal import get_model_arch



if __name__ == '__main__':
    batch_size: int = 16
    max_epochs: int = 100
    lr: float = 1e-4

    train_ids, validation_ids, test_ids = dataset.get_data_split_ids(json_path="/work3/s204135/data/raw/data.json")
    # ds = dataset.LANoireIndexDataset()
    ds = dataset.LANoireVideoDataset(json_path="/work3/s204135/data/raw/data.json", num_frames=8)
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)


    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # vid_embeddings = utils.load_pickle("video_embeddings.pkl")
    # model = video_encoder.VideoMLP(video_embeddings=vid_embeddings, hidden_size=512)
    model = video_encoder.VideoModel(lr=lr, model_name="facebook/timesformer-base-finetuned-k400")

    # model_arch = get_model_arch(model, (1,), dtypes=[torch.long])

    wandb_logger = utils.setup_logger(tags=["unimodal", "video", "end-to-end"], config={"lr": lr, "batch_size": batch_size})

    trainer = L.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
