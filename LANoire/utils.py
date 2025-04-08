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


def setup_logger(project: str = "LANoire", entity: str = "pydqn", tags: list[str] = ["unimodal", "text"], config: dict = {"lr": 0.0001, "batch_size": 100}, note: str = "Missing note"):
    logger = WandbLogger(project=project, entity=entity, tags=tags, config=config, notes=note)
    return logger
