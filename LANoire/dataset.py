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

        # Manual insertion of missing subjects
        # self.id_to_subject[16] = "elsa_lichtmann"

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx: int, debug: bool = False):
        return self.prepare_modalities(idx, debug)

    def prepare_modalities(self, idx: int, debug: bool = False):
        if not self.modalities:
            raise ValueError("Must have at least one modality")

        answer: dict = self.answers[f"a{idx + 1}"]
        question_id: int = answer["question id"]
        question: str = self.questions[f"q{question_id}"]["name"]
        subject_name: str = self.id_to_subject[answer["subject id"]]
        case_id: int = self.subjects[subject_name]["case id"]
        case: str = self.id_to_case[case_id]  # e.g. "16_the_naked_city"
        label: int = self.class_map[answer["class"]]

        directory: Path = Path(f"shortened_dataset/{case}/{subject_name}")
        filename: str = answer["name"].removesuffix(".mp3")

        subject_name: str = self.subjects[subject_name]["name"]

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
                    question_audio, _ = librosa.load(directory / f"{question}/{subject_name}_question_{question[1:]}.mp3")
                    answer_audio, _ = librosa.load(directory / f"{question}/{filename}.mp3")
                    data.append({"question": question_audio, "answer": answer_audio})
            elif modality == Modality.VIDEO:
                if debug:
                    video_dir = directory / f"original/{filename}"
                    
                    assert len(glob.glob(str(video_dir) + "*.png")) > 0, f"Missing video frames {video_dir}"
                    data.append(2)
                else:
                    video_frames = [np.asarray(Image.open(frame)) for frame in glob.glob(str(directory / f"original/{filename}*.png"))]
                    data.append(np.array(video_frames))

        data.append(label)
        return tuple(data)
    
if __name__ == '__main__':
    # dataset = LANoireDataset(modalities=(Modality.TEXT, Modality.AUDIO,))
    dataset = LANoireDataset(modalities=(Modality.AUDIO,))
    
    indices = [131, 132, 133, 134, 135, 136, 456, 457, 458, 459, 460, 461, 462]

    import tqdm
    # for idx in indices:
    #     text, audio, video, label = dataset.__getitem__(idx, debug=True)

    for i in tqdm.tqdm(range(len(dataset))):
        text, audio, label = dataset.__getitem__(i, debug=True)
