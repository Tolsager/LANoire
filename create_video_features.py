import os
from LANoire import video_encoder, dataset

import torch
import lightning as L

if __name__ == '__main__':
    if os.name == "nt":
        json_path = "data/raw/data.json"
    elif os.name == "posix":
        json_path = "/work3/s204135/data/raw/data.json"
    
    video_encoder = video_encoder.VideoEncoder(model_name="facebook/timesformer-base-finetuned-k400")

    ds = dataset.LANoireVideoDataset(json_path=json_path, bounding_boxes_path="bounding_boxes.pkl")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)

    trainer = L.Trainer()
    trainer.test(model=video_encoder, dataloaders=dataloader)
