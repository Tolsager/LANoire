import sys
sys.path.append("")
from LANoire import dataset, utils

import torch
import torch.nn as nn
from torchvision import models
import mediapipe as mp
import cv2
import numpy as np
from typing import List

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


def extract_face_mediapipe(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame_rgb)
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        return x1, y1, x2, y2
    return None


class FeatureExtractor(nn.Module): # MobileNetV2
    def __init__(self):
        super().__init__()
        base_model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class ClassifierHead(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.classifier(h_n)


def get_bounding_boxes(frames: List[np.ndarray]):
    bboxes = []
    for frame in frames:
        bbox_coordinates = extract_face_mediapipe(frame)
        bboxes.append(bbox_coordinates)

    return bboxes


if __name__ == "__main__":
    from tqdm import tqdm
    dataset = dataset.LANoireDataset(modalities=(dataset.Modality.VIDEO,))

    all_bboxes = {}

    for i in tqdm(range(len(dataset))):
        frames, _ = dataset[i]
        video_bboxes = get_bounding_boxes(frames)
        all_bboxes[i] = video_bboxes
    
    utils.save_pickle("bounding_boxes.pkl", all_bboxes)

