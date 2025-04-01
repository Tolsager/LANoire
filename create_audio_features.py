from LANoire.audio_encoder import CLAPDataModule, CLAPModel
import lightning as L

if __name__ == "__main__":
    model = CLAPModel()
    dm = CLAPDataModule(num_workers=0, persistent_workers=False)
    trainer = L.Trainer()
    batches = trainer.test(model, datamodule=dm)
