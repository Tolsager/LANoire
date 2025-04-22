from enum import IntEnum
from sklearn.model_selection import train_test_split
import cv2
import json
import glob
import torch
import torchaudio
import torchvision
from pathlib import Path

from transformers import AutoImageProcessor
from torch.utils.data import Dataset
from LANoire import utils

class Modality(IntEnum):
    TEXT = 1
    AUDIO = 2
    VIDEO = 3


# elsa_lichtman has two subject ids in the questions/answers but only one in the subject dict.


class LANoireDataset(Dataset):
    def __init__(self, json_path: str = "data/raw/data.json", data_dir: str = "data/raw", modalities: tuple[Modality]|None = (Modality.AUDIO,)):
        data_json = utils.load_json(json_path)

        self.data_dir = data_dir

        if modalities is not None:
            self.modalities = set(modalities)  # Store as a set for quick lookups
        self.cases = data_json["cases"][0]
        self.subjects = data_json["subjects"][0]
        self.questions = data_json["questions"][0]
        self.answers = data_json["answers"][0]

        self.class_map = {"lie": 0, "truth": 1}
        self.id_to_subject = {value["id"]: key for key, value in self.subjects.items()}
        self.id_to_case = {values["id"]: case for case, values in self.cases.items()}

        # Manual insertion of missing subjects
        # self.id_to_subject[16] = "elsa_lichtmann"

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx: int, debug: bool = False):
        return self.prepare_modalities(idx, debug)

    def prepare_modalities(self, idx: int, debug: bool = False):
        # if not self.modalities:
        #     raise ValueError("Must have at least one modality")

        answer: dict = self.answers[f"a{idx + 1}"]
        question_id: int = answer["question id"]
        question: str = self.questions[f"q{question_id}"]["name"]
        subject_name: str = self.id_to_subject[answer["subject id"]]
        case_id: int = self.subjects[subject_name]["case id"]
        case: str = self.id_to_case[case_id]  # e.g. "16_the_naked_city"
        label: int = self.class_map[answer["class"]]

        directory: Path = Path(f"{self.data_dir}/{case}/{subject_name}")
        filename: str = answer["name"].removesuffix(".mp3")

        subject_name: str = self.subjects[subject_name]["name"]

        if self.modalities is None:
            return {"answer": answer, "question": question, "subject": subject_name, "case": case, "label": label}

        data = []

        for modality in self.modalities:
            if modality == Modality.TEXT:
                answer_text: str = answer["text"]
                question_text: str = self.questions[f"q{question_id}"]["text"]
                data.append({"q": question_text, "a": answer_text})
            elif modality == Modality.AUDIO:
                if debug:
                    question_dir = directory / f"{question}/{subject_name}_question_{question[1:]}.mp3"
                    answer_dir = directory / f"{question}/{filename}.mp3"
                    assert (question_dir).exists(), f"Invalid directory: {question_dir}"
                    assert (answer_dir).exists(), f"Invalid directory: {answer_dir}"
                    data.append({"question": str(question_dir), "answer": str(answer_dir)})
                else:
                    question_audio, _ = torchaudio.load(directory / f"{question}/{subject_name}_question_{question[1:]}.mp3")
                    answer_audio, _ = torchaudio.load(directory / f"{question}/{filename}.mp3")

                    if len(question_audio.shape) > 1:
                        question_audio = question_audio.mean(dim=0)
                    if len(answer_audio.shape) > 1:
                        answer_audio = answer_audio.mean(dim=0)
                    
                    data.append({"question": question_audio, "answer": answer_audio})
            elif modality == Modality.VIDEO:
                if debug:
                    video_dir = directory / f"original/{filename}"
                    
                    assert len(glob.glob(str(video_dir) + "*.png")) > 0, f"Missing video frames {video_dir}"
                    data.append(2)
                else:
                    video_frames = [cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB) for frame in glob.glob(str(directory / f"original/{filename}*.png"))] # Loads the images as BGR
                    data.append(video_frames)

        data.append(label)
        return tuple(data)

def get_data_split_ids(json_path: str = "data/raw/data.json") -> tuple[list[int], list[int], list[int]]: 
    data_json = utils.load_json(json_path)
    
    answers = data_json["answers"][0]
    ids = []
    labels = []
    class_map = {"lie": 0, "truth": 1}
    for d in answers.values():
        ids.append(d["id"]-1)
        labels.append(class_map[d["class"]])

    train_ids, test_ids, _, test_labels = train_test_split(ids, labels, test_size=0.2, stratify=labels, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, stratify=test_labels, random_state=42)

    return train_ids, val_ids, test_ids
    

class LANoireIndexDataset(Dataset):
    """
    Used to get indecies to lookup embeddings and the corresponding label of the statement
    """
    def __init__(self, json_path: str = "data/raw/data.json"):
        data_json = utils.load_json(json_path)
        
        self.answers = data_json["answers"][0]
        self.class_map = {"lie": 0, "truth": 1}

    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, idx):
        answer = self.answers[f"a{idx+1}"]
        id = torch.tensor(answer["id"] - 1)
        label = torch.tensor(self.class_map[answer["class"]], dtype=torch.float32)
        return id, label


class LANoireVideoDataset(Dataset):
    def __init__(self, json_path: str = "data/raw/data.json", bounding_boxes_path: str = "bounding_boxes.pkl", num_frames: int = 8):
        data_json = utils.load_json(json_path)
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.bounding_boxes = utils.load_pickle(bounding_boxes_path)
        self.answers = data_json["answers"][0]
        self.num_frames = 8

    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, idx):
        frames = self.bounding_boxes[idx]
        frames = list(filter(lambda x: x is not None, frames))

        frames = frames[:self.num_frames]
        if len(frames) < 8:
            frames.extend(frames[-1]*(self.num_frames - len(frames)))

        pixel_values = self.image_processor(frames, return_tensors="pt")

        return idx, pixel_values


if __name__ == '__main__':
    ids = get_data_split_ids()
    dataset = LANoireDataset(modalities=(Modality.VIDEO,))

    import tqdm

    for i in tqdm.tqdm(range(len(dataset))):
        audio, label = dataset.__getitem__(i, debug=True)
