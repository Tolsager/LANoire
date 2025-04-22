from tqdm import tqdm
from LANoire import dataset, video_encoder, utils

if __name__ == '__main__':
    dataset = dataset.LANoireDataset(modalities=(dataset.Modality.VIDEO,))

    all_bboxes = {}

    for i in tqdm(range(len(dataset))):
        frames, _ = dataset[i]
        video_bboxes = video_encoder.get_bounding_boxes(frames)
        all_bboxes[i] = video_bboxes

    utils.save_pickle("bounding_boxes.pkl", all_bboxes)