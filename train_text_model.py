from LANoire import dataset, text_encoder, utils
from torch.utils.data import Subset, DataLoader
import torch
from LANoire.utils import get_model_arch
import lightning as L
import wandb

if __name__ == "__main__":
    debug = False
    train_ids, validation_ids, test_ids = dataset.get_data_split_ids()
    ds = dataset.LANoireIndexDataset()
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)
    train_dataloader = DataLoader(
        train_set, batch_size=48, shuffle=True, num_workers=3, persistent_workers=True
    )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=48,
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(test_set, batch_size=48, shuffle=False, num_workers=0)

    max_epochs = 400

    # model_name = "distilbert"
    # embeds_name = "distilbert_embeds.pkl"
    # hidden_size = 512

    # model_name = "bert-base-uncased"
    # embeds_name = "bert_base_uncased_embeds.pkl"
    # hidden_size = 768

    model_name = "roberta-base"
    embeds_name = "bert_base_uncased_embeds.pkl"
    hidden_size = 768

    embeds = utils.load_pickle(f"data/processed/{embeds_name}")
    model = text_encoder.TextMLP(embeds=embeds, hidden_size=hidden_size)

    model_arch = get_model_arch(model, (1,), dtypes=[torch.long])

    wandb.finish()
    wandb_logger = utils.setup_logger(
        tags=["unimodal", "text", model_name],
        config={"lr": 1e-4, "batch_size": 48},
        note=model_arch,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        "models", monitor="val_acc", mode="max"
    )
    trainer = L.Trainer(
        max_epochs=max_epochs, logger=wandb_logger if not debug else None, callbacks=[checkpoint_callback]
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )
    trainer.test(dataloaders=test_dataloader, ckpt_path="best")
