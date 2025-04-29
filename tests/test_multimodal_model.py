from LANoire.multimodal_model import TextAudioCAF, TextAudioVideo
from LANoire.dataset import BiModalityDm, AllModalityDm
import lightning as L


def test_TextAudioCAF():
    model = TextAudioCAF(batch_size=2)
    dm = BiModalityDm(batch_size=2)
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=dm)

def test_AllModalityDm():
    dm = AllModalityDm(batch_size=2, drop_last=True)
    trainer = L.Trainer(fast_dev_run=True)

    model = TextAudioVideo(batch_size=2, feature_fusion="CAF")
    trainer.fit(model, datamodule=dm)

    model = TextAudioVideo(batch_size=2, feature_fusion="GMU")
    trainer.fit(model, datamodule=dm)

    model = TextAudioVideo(batch_size=2, feature_fusion="CAT")
    trainer.fit(model, datamodule=dm)