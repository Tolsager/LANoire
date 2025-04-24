from LANoire import utils
from LANoire.audio_encoder import ClapEeDm, CLAPModelEE
import lightning as L

if __name__ == "__main__":
    batch_size = 48
    num_workers = 7
    # batch_size = 2
    # num_workers = 0
    max_epochs = 200
    lr = 1e-4
    dropout = 0.3

    dm = ClapEeDm(train_batch_size=batch_size, num_workers=num_workers)
    model = CLAPModelEE(dropout=dropout)
    wandb_logger = utils.setup_logger(
        tags=["unimodal", "audio", "CLAP", "end-to-end"],
        config={"lr": lr, "batch_size": batch_size, "dropout": dropout},
    )

    trainer = L.Trainer(
        max_epochs=max_epochs, logger=wandb_logger,
    )
    trainer.fit(
        model=model,
        datamodule=dm
    )
