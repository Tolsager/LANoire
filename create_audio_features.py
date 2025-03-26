from LANoire.audio_encoder import CLAPDataModule, CLAPModel, dict_CLAP_out
from utils import save_pickle
import lightning as L

if __name__ == "__main__":
    model = CLAPModel()
    dm = CLAPDataModule()
    trainer = L.Trainer()
    batches = trainer.predict(model, datamodule=dm)
    CLAP_embeds = dict_CLAP_out(batches)
    save_pickle("CLAP_embeds.pkl", CLAP_embeds)
