from LANoire.audio_encoder import WhisperAudioDS


def test_whisper_audio_ds():
    ds = WhisperAudioDS()
    s = ds[0]
    assert s.shape == (1, 80, 3000)
