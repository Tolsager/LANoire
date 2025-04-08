from LANoire.unimodal_model import ClapDm, ClapMlp
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
    

if __name__ == '__main__':
    load_dotenv()
    dm = ClapDm(train_batch_size=100)
    model = ClapMlp()
    model_arch = get_model_arch(model, input_size=(1,), dtypes=[torch.long])
    # wandb_logger = L.pytorch.loggers.WandbLogger(project="LANoire", entity="pydqn", notes=model_arch)

    # trainer = L.Trainer(max_epochs=50, logger=wandb_logger)
    trainer = L.Trainer(max_epochs=50)
    trainer.fit(model, datamodule=dm)