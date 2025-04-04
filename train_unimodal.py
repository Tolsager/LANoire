from LANoire.unimodal_model import ClapDm, ClapMlp
import lightning as L

dm = ClapDm(train_batch_size=2)
model = ClapMlp()
trainer = L.Trainer(fast_dev_run=True)
trainer.fit(model, datamodule=dm)