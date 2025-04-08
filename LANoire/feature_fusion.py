from torch import nn
import torch
import torch.nn.functional as F

class CA_SA(nn.Module):
    def __init__(self, dim=32):
        super(CA_SA, self).__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.Q = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, feat1, feat2):
        Q = self.Q(feat1)
        K = self.K(feat2)
        V = self.V(feat2)
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.attend(dots)
        out = torch.bmm(attn, V)
        return out

class CAF(nn.Module):
    """Cross Attention Fusion Module
    This module takes two or three features and fuses them using cross attention.
    The output is a single feature that is the result of the fusion.
    Assumes features are of shape (b, 512)
    """
    def __init__(self, n_features: int=2, attention_dim: int=32):
        super().__init__()
        self.attention_dim = attention_dim
        self.SA = CA_SA(dim=attention_dim)
        self.CAs = [CA_SA(dim=attention_dim) for _ in range(n_features)]

        self.n_features = n_features
    
    def forward(self, *features):
        # all features should be of shape (b, 512)
        b = features[0].shape[0]
        F_U = torch.cat([F.normalize(f, dim=-1) for f in features], dim=-1)
        F_U = F_U.view(b, -1, self.attention_dim)
        feat_Z = [self.CAs[i](F_U, f.view(b, -1, self.attention_dim)) for i, f in enumerate(features)]

        feat = sum(feat_Z)
        feat = self.SA(feat, feat) + feat
        feat = feat.chunk(self.n_features, dim=1)
        feat = sum([f.view(b, -1) for f in feat])
        return feat
