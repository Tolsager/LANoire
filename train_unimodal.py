from LANoire.unimodal_model import ClapDm, ClapMlp, WhisperMlp
import os 
import torch
import lightning as L
from dotenv import load_dotenv
import torchinfo

def get_model_arch(model: torch.nn.Module, input_size: tuple[int], dtypes:tuple[torch.dtype]) -> str:
    s = torchinfo.summary(
        model, input_size=input_size, dtypes=dtypes, verbose=0
    )
    def repr(self):
        divider = "=" * self.formatting.get_total_width()
        all_layers = self.formatting.layers_to_str(self.summary_list, self.total_params)
        summary_str = (
            f"{divider}\n"
            f"{self.formatting.header_row()}{divider}\n"
            f"{all_layers}{divider}\n"
        )
        return summary_str
    s.__repr__ = repr.__get__(s)
    return repr(s)
    
# debug
# os.environ["WANDB_MODE"] = "offline"
# enable_checkpointing = False
# max_epochs = 10

enable_checkpointing = True
max_epochs = 50
batch_size = 10
lr = 1e-3
dropout = 0.3
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint("models", monitor="val_acc", mode="max")
dm = ClapDm(train_batch_size=batch_size)
# tags = ["CLAP", "unimodal", "audio"]
tags = ["Whisper", "unimodal", "audio"]
# model = ClapMlp(lr=lr, dropout=dropout)
model = WhisperMlp(lr=lr, dropout=dropout)
model_arch = get_model_arch(model, input_size=(1,), dtypes=[torch.long])
wandb_logger = L.pytorch.loggers.WandbLogger(project="LANoire", entity="pydqn", notes=model_arch, config={"lr": lr, "batch_size": batch_size, "dropout": dropout})
trainer = L.Trainer(max_epochs=max_epochs, logger=wandb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=dm)
