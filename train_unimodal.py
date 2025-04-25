from LANoire.unimodal_model import EmbeddingDm, ClapMlp, WhisperMlp, Wav2vec2Mlp
import os
import torch
import lightning as L
from dotenv import load_dotenv

from LANoire.utils import get_model_arch

# debug
# os.environ["WANDB_MODE"] = "offline"
# enable_checkpointing = False
# max_epochs = 10

if __name__ == "__main__":
    enable_checkpointing = True
    max_epochs = 50
    batch_size = 50
    lr = 1e-3
    dropout = 0.3
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        "models", monitor="val_acc", mode="max"
    )
    dm = EmbeddingDm(train_batch_size=batch_size)
    # CLAP
    # tags = ["CLAP", "unimodal", "audio"]
    # model = ClapMlp(lr=lr, dropout=dropout)

    # Whisper
    # tags = ["Whisper", "unimodal", "audio"]
    # model = WhisperMlp(lr=lr, dropout=dropout)

    # wav2vec2
    tags = ["wav2vec2", "unimodal", "audio"]
    model = Wav2vec2Mlp(lr=lr, dropout=dropout)

    model_arch = get_model_arch(model, input_size=(1,), dtypes=[torch.long])
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="LANoire",
        entity="pydqn",
        notes=model_arch,
        config={"lr": lr, "batch_size": batch_size, "dropout": dropout},
    )
    trainer = L.Trainer(
        max_epochs=max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback]
    )
    trainer.fit(model, datamodule=dm)
