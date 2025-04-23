from LANoire import dataset, text_encoder, utils
from torch.utils.data import Subset, DataLoader
import lightning as L

if __name__ == "__main__":
    batch_size = 48
    train_ids, validation_ids, test_ids = dataset.get_data_split_ids()
    ds = dataset.LANoireDataset(modalities=[dataset.Modality.TEXT])
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=3, persistent_workers=True
    )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3)

    max_epochs = 200

    model_name = "roberta-base"
    model = text_encoder.TransEE()

    wandb_logger = utils.setup_logger(
        tags=["unimodal", "text", model_name, "end-to-end"],
        config={"lr": 1e-4, "batch_size": batch_size},
    )

    trainer = L.Trainer(
        max_epochs=max_epochs, logger=wandb_logger, fast_dev_run=True
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )
