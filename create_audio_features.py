from LANoire.audio_encoder import CLAPDataModule, CLAPModel, WhisperModel, WhisperDataModule, Wav2Vec2Model, Wav2Vec2DataModule
import lightning as L

if __name__ == "__main__":
    # CLAP
    # model = CLAPModel()
    # dm = CLAPDataModule(num_workers=0, persistent_workers=False)
    # trainer = L.Trainer()
    # batches = trainer.test(model, datamodule=dm)
    
    # Whisper
    # model = WhisperModel()
    # dm = WhisperDataModule(num_workers=0)
    # trainer = L.Trainer()
    # batches = trainer.test(model, datamodule=dm)

    # wav2vec2
    model = Wav2Vec2Model()
    dm = Wav2Vec2DataModule()
    trainer = L.Trainer()
    batches = trainer.test(model, datamodule=dm)