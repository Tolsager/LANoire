import LANoire.video_encoder
import LANoire.dataset
import LANoire.utils

import torch
import lightning as L

if __name__ == '__main__':
    video_encoder = LANoire.video_encoder.VideoEncoder()

    ds = LANoire.dataset.LANoireVideoDataset(json_path="data/raw/data.json", bounding_boxes_path="bounding_boxes.pkl")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)

    trainer = L.Trainer()
    trainer.test(model=video_encoder, dataloaders=dataloader)

    # video_embeddings = {}
    # for i in tqdm(range(len(ds))):
    #     bboxes = bounding_boxes[i]
    #     if len(bboxes) == 0:
    #         result = None
    #     else: # Otherwise take the first 8 frames and get the embedding of it
    #         video_frames, _ = ds[i]
    #         frames = video_frames[:num_frames]
    #         frames = [img[y1:y2, x1:x2] for img, (x1, y1, x2, y2) in zip(frames, bboxes[:num_frames])]
    #         # for k in range(0, len(video_frames) - num_frames + 1, num_frames):
    #         #     frames = video_frames[k:k+num_frames]
    #         #     bboxes_slice = bboxes[k:k+num_frames]
    #         #     if len(frames) < num_frames:
    #         #         frames.extend([frames[-1]]*(num_frames - len(frames)))
    #         #         bboxes_slice.extend([bboxes_slice[-1]]*(num_frames - len(frames)))

                
    #         output = video_encoder(frames)

    #         result = output[:, 0, :] # CLS Token embedding

    #     video_embeddings[i] = result

    # LANoire.utils.save_pickle("video_embeddings.pkl", video_embeddings)
        