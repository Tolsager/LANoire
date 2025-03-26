from enum import IntEnum
import json
import glob
import numpy as np
import librosa
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class Modality(IntEnum):
    TEXT = 1
    AUDIO = 2
    VIDEO = 3


# elsa_lichtman has two subject ids in the questions/answers but only one in the subject dict.


class LANoireDataset(Dataset):
    def __init__(self, json_path: str = "data/data.json", modalities: tuple[Modality] = (Modality.AUDIO,)):
        with open(json_path, "r") as f:
            data_json = json.load(f)

        self.modalities = set(modalities)  # Store as a set for quick lookups
        self.cases = data_json["cases"][0]
        self.subjects = data_json["subjects"][0]
        self.questions = data_json["questions"][0]
        self.answers = data_json["answers"][0]

        self.class_map = {"lie": 0, "truth": 1}
        self.id_to_subject = {value["id"]: key for key, value in self.subjects.items()}
        self.id_to_case = {values["id"]: case for case, values in self.cases.items()}

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx: int):
        return self.prepare_modalities(idx)

    def prepare_modalities(self, idx: int):
        if not self.modalities:
            raise ValueError("Must have at least one modality")

        answer: dict = self.answers[f"a{idx + 1}"]
        question_id: int = answer["question id"]
        question: str = self.questions[f"q{question_id}"]["name"]
        subject_name: str = self.id_to_subject[answer["subject id"]]
        case_id: int = self.subjects[subject_name]["case id"]
        case: str = self.id_to_case[case_id]  # e.g., "16_the_naked_city"
        label: int = self.class_map[answer["class"]]

        directory: Path = Path(f"shortened_dataset/{case}/{subject_name}")
        filename: str = answer["name"].removesuffix(".mp3")

        data = []

        for modality in self.modalities:
            if modality == Modality.TEXT:
                answer_text: str = answer["text"]
                question_text: str = self.questions[f"q{question_id}"]["text"]
                data.append({"q": question_text, "a": answer_text})
            elif modality == Modality.AUDIO:
                question_audio, _ = librosa.load(directory / f"{question}/{subject_name}_question_{question[1:]}.mp3")
                answer_audio, _ = librosa.load(directory / f"{question}/{filename}.mp3")
                data.append({"question": question_audio, "answer": answer_audio})
            elif modality == Modality.VIDEO:
                video_frames = [np.asarray(Image.open(frame)) for frame in glob.glob(str(directory / f"original/{filename}*.png"))]
                data.append(np.array(video_frames))

        data.append(label)
        return tuple(data)
    
if __name__ == '__main__':
    dataset = LANoireDataset(modalities=(Modality.TEXT,))
    

    import tqdm
    for i in tqdm.tqdm(range(len(dataset))):
        text, label = dataset[i]
