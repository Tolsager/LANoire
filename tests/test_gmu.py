from LANoire.gmu import BimodalGMU, TrimodalGMU
import torch

def test_bimodalgmu():
    gmu = BimodalGMU(200, 300)
    f1 = torch.rand((1, 200))
    f2 = torch.rand((1, 300))
    out = gmu(f1, f2)
    assert out.shape == (1,)

def test_trimodalgmu():
    gmu = TrimodalGMU(200, 300, 150)
    f1 = torch.rand((1, 200))
    f2 = torch.rand((1, 300))
    f3 = torch.rand((1, 150))
    out = gmu(f1, f2, f3)
    assert out.shape == (1,)