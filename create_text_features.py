from LANoire.text_encoder import TextEncoder
from LANoire.dataset import LANoireDataset, Modality
from torch.utils.data import DataLoader
import lightning as L


if __name__ == '__main__':
    dataset = LANoireDataset(modalities=(Modality.TEXT,))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    # model = TextEncoder("bert-base-uncased", "bert_base_uncased_embeds.pkl")
    model = TextEncoder("roberta-base", "roberta_base_embeds.pkl")
    # model = TextEncoder("distilbert-base-uncased", "distilbert_embeds.pkl")
    trainer = L.Trainer()
    trainer.test(model=model, dataloaders=dataloader)
