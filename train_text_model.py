from LANoire import dataset, text_encoder, utils
from torch.utils.data import Subset
import lightning as L

if __name__ == '__main__':
    train_ids, validation_ids, test_ids = dataset.get_data_split_ids()
    ds = dataset.LANoireIndexDataset()
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)

    trainer = L.Trainer()