import pickle  # noqa: I001
import typing
import numpy as np
import numpy.typing as npt


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    return data


def save_pickle(filepath: str, obj: typing.Any):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
