from LANoire.multimodal_model import TextAudioCat, TextAudioCAF
from LANoire.unimodal_model import EmbeddingDm
from LANoire.utils import get_model_arch
import os
import torch
import lightning as L
from dotenv import load_dotenv




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
    dm = EmbeddingDm(train_batch_size=batch_size)

    # Text + Audio
    tags = ["CLAP", "roberta", "bimodal", "audio", "text"]
    model = TextAudioCAF(lr=lr, dropout=dropout, weight_decay=weight_decay)

    model_arch = get_model_arch(model, input_size=(1,), dtypes=[torch.long])
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="LANoire",
        entity="pydqn",
        notes=model_arch,
        config={"lr": lr, "batch_size": batch_size, "dropout": dropout, "weight_decay": weight_decay},
        tags=tags
    )
    trainer = L.Trainer(
        max_epochs=max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback], fast_dev_run=False
    )
    trainer.fit(model, datamodule=dm)
