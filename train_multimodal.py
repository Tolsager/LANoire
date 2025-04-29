from LANoire.multimodal_model import (
    TextAudioCat,
    TextAudioCAF,
    TextAudioGmu,
    TextVideoGmu,
    TextVideoCat,
    TextVideoCaf,
    AudioVideoCaf,
    AudioVideoCat,
    AudioVideoGmu,
    AllCat,
    AllCaf,
    AllGmu,
    TextAudioVideo,
)
from LANoire.unimodal_model import EmbeddingDm
from LANoire.utils import get_model_arch
from LANoire.dataset import AllModalityDm, LANoireIndexDataset
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
    batch_size = 64
    lr = 1e-4
    dropout = 0.2
    weight_decay = 0.1
    w_contr = 0.1
    temperature = 0.5
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        "models", monitor="val_acc", mode="max"
    )
    # dm = AllModalityDm(batch_size=batch_size)
    dm = EmbeddingDm(batch_size=batch_size)
    # dm = LANoireIndexDataset(batch_size=batch_size)

    # Text + Audio
    tags = ["CLAP", "roberta", "bimodal", "audio", "text", "CAF"]
    model = TextAudioCAF(
        lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size, w_contr=w_contr, temperature=temperature
    )

    # tags = ["CLAP", "roberta", "bimodal", "audio", "text", "gmu"]
    # model = TextAudioGmu(lr=lr, dropout=dropout, weight_decay=weight_decay)

    # Text + Video
    # tags = ["videomae", "bimodal", "video", "text", "gmu", "roberta"]
    # model = TextVideoGmu(lr=lr, dropout=dropout, weight_decay=weight_decay)

    # tags = ["roberta", "videomae", "bimodal", "video", "text", "cat"]
    # model = TextVideoCat(lr=lr, dropout=dropout, weight_decay=weight_decay)

    # tags = ["roberta", "videomae", "bimodal", "video", "text", "caf"]
    # model = TextVideoCaf(lr=lr, dropout=dropout, weight_decay=weight_decay)

    # Audio + Video
    # tags = ["videomae", "bimodal", "video", "audio", "caf", "CLAP"]
    # model = AudioVideoCaf(lr=lr, dropout=dropout, weight_decay=weight_decay)

    # tags = ["videomae", "bimodal", "video", "audio", "cat", "CLAP"]
    # model = AudioVideoCat(lr=lr, dropout=dropout, weight_decay=weight_decay)

    # tags = ["videomae", "bimodal", "video", "audio", "gmu", "CLAP"]
    # model = AudioVideoGmu(lr=lr, dropout=dropout, weight_decay=weight_decay)

    # All
    # tags = ["videomae", "trimodal", "cat", "CLAP", "roberta"]
    # model = AllCat(dropout=dropout, weight_decay=weight_decay)

    # tags = ["videomae", "trimodal", "caf", "CLAP", "roberta"]
    # model = AllCaf(dropout=dropout, weight_decay=weight_decay)

    # tags = ["videomae", "trimodal", "concat", "CLAP", "roberta"]
    # model = TextAudioVideo(feature_fusion="GMU")

    # model_arch = get_model_arch(model, input_size=(1,), dtypes=[torch.long])
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="LANoire",
        entity="pydqn",
        config={"lr": lr, "batch_size": batch_size, "w_contr": w_contr, "temperature": temperature},
        tags=tags,
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=dm)
