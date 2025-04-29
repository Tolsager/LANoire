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

class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, feat_2 as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        def l_ij(i, j):
            # z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            # if self.verbose: print("sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = (
                torch.ones((2 * self.batch_size,))
                .to(emb_i.device)
                .scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            )
            # if self.verbose: print("1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            # if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            # if self.verbose: print("loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss

class CAF(nn.Module):
    """Cross Attention Fusion Module
    This module takes two or three features and fuses them using cross attention.
    The output is a single feature that is the result of the fusion.
    Assumes features are of shape (b, 512)
    """

    def __init__(self, n_features: int = 2, attention_dim: int = 32):
        super().__init__()
        self.attention_dim = attention_dim
        self.SA = CA_SA(dim=attention_dim)
        self.CAs = nn.ModuleList([CA_SA(dim=attention_dim) for _ in range(n_features)])
        self.n_features = n_features

    def forward(self, *features):
        # all features should be of shape (b, 512)
        b = features[0].shape[0]
        F_U = torch.cat([F.normalize(f, dim=-1) for f in features], dim=-1)
        F_U = F_U.view(b, -1, self.attention_dim)
       
        feat_Z = sum([self.CAs[i](F_U, f.view(b, -1, self.attention_dim)) for i, f in enumerate(features)])

        feat = feat_Z
        feat = self.SA(feat, feat) + feat
        feat = feat.chunk(self.n_features, dim=1)
        feat = sum([f.view(b, -1) for f in feat])
        return feat
