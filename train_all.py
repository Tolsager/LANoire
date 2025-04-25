from LANoire.dataset import AllModalityDm
from LANoire.multimodal_model import TextAudioVideo
import lightning as L

if __name__ == "__main__":
    # debug
    # os.environ["WANDB_MODE"] = "offline"
    # enable_checkpointing = False
    # max_epochs = 10
    enable_checkpointing = False
    max_epochs = 200
    batch_size = 48
    lr = 1e-4
    dropout = 0.2
    weight_decay = 0.1
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        "models", monitor="val_acc", mode="max"
    )
    dm = AllModalityDm(train_batch_size=batch_size)

    tags = ["videomae", "trimodal", "gmu", "CLAP", "roberta"]
    model = TextAudioVideo(dropout=dropout, weight_decay=weight_decay)

    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="LANoire",
        entity="pydqn",
        config={"lr": lr, "batch_size": batch_size, "dropout": dropout, "weight_decay": weight_decay},
        tags=tags
    )
    trainer = L.Trainer(
        max_epochs=max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback], fast_dev_run=True
    )
    trainer.fit(model, datamodule=dm)