from LANoire import dataset, text_encoder, utils
from torch.utils.data import Subset, DataLoader
import torch
from train_unimodal import get_model_arch
import lightning as L

if __name__ == '__main__':
    debug = False
    train_ids, validation_ids, test_ids = dataset.get_data_split_ids()
    ds = dataset.LANoireIndexDataset()
    train_set = Subset(ds, indices=train_ids)
    validation_set = Subset(ds, indices=validation_ids)
    test_set = Subset(ds, indices=test_ids)
    train_dataloader = DataLoader(train_set, batch_size=48, shuffle=True, num_workers=3, persistent_workers=True)
    validation_dataloader = DataLoader(validation_set, batch_size=48, shuffle=False, num_workers=3, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=48, shuffle=False, num_workers=3)

    distilbert_embeddings = utils.load_pickle("distilbert_embeds.pkl")
    model = text_encoder.TextMLP(embeds=distilbert_embeddings, hidden_size=512)

    model_arch = get_model_arch(model, (1,), dtypes=[torch.long])

    wandb_logger = utils.setup_logger(tags=["unimodal", "text"], config={"lr": 1e-4, "batch_size": 48}, note=model_arch)

    trainer = L.Trainer(max_epochs=150, logger=wandb_logger if not debug else None)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
