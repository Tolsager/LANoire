from LANoire.audio_encoder import WhisperAudioDS, Wav2Vec2DS, Wav2Vec2Model, ClapEeDs
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
    
def test_ClapEeDs():
    ds = ClapEeDs()
    features, label = ds[0]
    input_features = features["input_features"]
    is_longer = features["is_longer"]
    assert input_features.shape == (4, 1001, 64)
    assert is_longer.shape == (1,)
