from LANoire.dataset import get_data_split_ids, AllModalityDs, AllModalityDm, BiModalityDs, BiModalityDm


def test_get_data_split_ids():
    train_ids, val_ids, test_ids = get_data_split_ids()

    assert len(train_ids) == 520, f"Expected 520 training IDs, got {len(train_ids)}"
    assert len(val_ids) == 65, f"Expected 65 validation IDs, got {len(val_ids)}"
    assert len(test_ids) == 65, f"Expected 65 test IDs, got {len(test_ids)}"

    assert (
        len(set(train_ids).intersection(set(val_ids))) == 0
    ), "Train and validation sets should not overlap"
    assert (
        len(set(train_ids).intersection(set(test_ids))) == 0
    ), "Train and test sets should not overlap"
    assert (
        len(set(val_ids).intersection(set(test_ids))) == 0
    ), "Validation and test sets should not overlap"


def test_AllModalityDs():
    ds = AllModalityDs()
    s = ds[0]
    assert len(s) == 4


def test_AllModalityDm():
    dm = AllModalityDm(batch_size=2)
    dm.setup("train")
    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 4

def test_BiModalityDs():
    ds = BiModalityDs()
    s = ds[0]
    assert len(s) == 3


def test_BiModalityDm():
    dm = BiModalityDm(batch_size=2)
    dm.setup("train")
    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 3