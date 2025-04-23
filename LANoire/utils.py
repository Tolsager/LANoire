import json
import torch
import torchinfo
import pickle  # noqa: I001
import typing
import numpy as np
import numpy.typing as npt

from lightning.pytorch.loggers import WandbLogger


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    return data


def save_pickle(filepath: str, obj: typing.Any):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_json(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def setup_logger(project: str = "LANoire", entity: str = "pydqn", tags: list[str] = ["unimodal", "text"], config: dict = {"lr": 0.0001, "batch_size": 100}, note: str = "Missing note"):
    logger = WandbLogger(project=project, entity=entity, tags=tags, config=config, notes=note)
    return logger

def get_model_arch(
    model: torch.nn.Module, input_size: tuple[int], dtypes: tuple[torch.dtype]
) -> str:
    s = torchinfo.summary(model, input_size=input_size, dtypes=dtypes, verbose=0)

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