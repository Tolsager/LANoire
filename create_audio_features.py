from LANoire.audio_encoder import CLAPDataModule, CLAPModel, WhisperModel, WhisperDataModule
import lightning as L

if __name__ == "__main__":
    # CLAP
    # model = CLAPModel()
    # dm = CLAPDataModule(num_workers=0, persistent_workers=False)
    # trainer = L.Trainer()
    # batches = trainer.test(model, datamodule=dm)
    
    # Whisper
    model = WhisperModel()
    dm = WhisperDataModule(num_workers=0)
    trainer = L.Trainer()
    batches = trainer.test(model, datamodule=dm)