from LANoire import dataset, text_encoder, utils
from torch.utils.data import Subset, DataLoader
import lightning as L

if __name__ == "__main__":
    batch_size = 32
    lr = 5e-5
    max_epochs = 50
    train_ids, validation_ids, test_ids = dataset.get_data_split_ids("/work3/s204135/data/raw/data.json")
    ds = dataset.LANoireDataset(json_path="/work3/s204135/data/raw/data.json", data_dir="/work3/s204135/data/raw",modalities=[dataset.Modality.TEXT])
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=7, persistent_workers=True
    )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    model_name = "roberta-base"
    model = text_encoder.TransEE(lr=lr)

    wandb_logger = utils.setup_logger(
        tags=["unimodal", "text", model_name, "end-to-end"],
        config={"lr": lr, "batch_size": batch_size},
    )

    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        "/work3/s204135/models", monitor="val_acc", mode="max", filename="roberta_EE_{epoch}"
    )
    trainer = L.Trainer(
        max_epochs=max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback],fast_dev_run=False
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )
    trainer.test(dataloaders=test_dataloader)
