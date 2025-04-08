from LANoire.feature_fusion import CAF
import torch

def test_CAF():
    # Create a CAF instance
    caf = CAF(n_features=2, attention_dim=32)

    # Create dummy input features
    feature1 = torch.randn(1, 512)  # Batch size of 1, feature size of 512
    feature2 = torch.randn(1, 512)

    # Forward pass through the CAF module
    output = caf(feature1, feature2)

    # Check the output shape
    assert output.shape == (1, 512), f"Expected output shape (1, 512), but got {output.shape}"

    # Create a CAF instance
    caf = CAF(n_features=3, attention_dim=32)

    # Create dummy input features
    feature1 = torch.randn(1, 512)  # Batch size of 1, feature size of 512
    feature2 = torch.randn(1, 512)
    feature3 = torch.randn(1, 512)

    # Forward pass through the CAF module
    output = caf(feature1, feature2, feature3)

    # Check the output shape
    assert output.shape == (1, 512), f"Expected output shape (1, 512), but got {output.shape}"