from LANoire.multimodal_model import TextAudioCAF
from LANoire.dataset import BiModalityDm
import lightning as L


def test_TextAudioCAF():
    model = TextAudioCAF(batch_size=2)
    dm = BiModalityDm(batch_size=2)
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=dm)

    