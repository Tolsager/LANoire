from LANoire.audio_encoder import WhisperAudioDS, Wav2Vec2DS, Wav2Vec2Model
import torch

def test_wav2vec2ds():
    ds = Wav2Vec2DS()
    s = ds[0]
    assert s.shape == (64_000,)

def test_whisper_audio_ds():
    ds = WhisperAudioDS()
    s = ds[0]
    assert s.shape == (1, 80, 3000)

def test_wav2vec2model():
    model = Wav2Vec2Model()
    input = torch.rand(1, 64000)
    out = model(input)
    assert out.shape == (1, 768)