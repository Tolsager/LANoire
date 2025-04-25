import os
from LANoire import video_encoder, dataset

import torch
import lightning as L

if __name__ == '__main__':
    if os.name == "nt":
        json_path = "data/raw/data.json"
    elif os.name == "posix":
        json_path = "/work3/s204135/data/raw/data.json"
    
    model_name, num_frames = "MCG-NJU/videomae-base-finetuned-kinetics", 16
    # model_name, num_frames = "facebook/timesformer-base-finetuned-k400", 8
    video_encoder = video_encoder.VideoEncoder(model_name=model_name)

    ds = dataset.LANoireVideoDataset(json_path=json_path, feature_extraction_level="bounding_box", num_frames=num_frames)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)

    trainer = L.Trainer()
    trainer.test(model=video_encoder, dataloaders=dataloader)
