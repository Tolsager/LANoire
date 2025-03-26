import pickle
import typing

def save_pickle(filepath: str, obj: typing.Any):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> typing.Any:
    with open(filepath, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    return data