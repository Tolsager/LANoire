from LANoire.audio_encoder import get_whisper_embeddings
import librosa

def test_get_whisper_embeddings():
    audio = librosa.ex("libri1")
    