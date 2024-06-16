import torch
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm


def cart2hyper_sph(vec):
    if not isinstance(vec, torch.Tensor):
        raise ValueError("invalid vec type: {}".format(type(vec)))
    r = torch.linalg.norm(vec, dim=-1)
    thetas = torch.zeros_like(vec)

    for i in range(1, vec.shape[1]):
        temp = torch.sqrt(torch.sum(vec[:, i - 1:] ** 2, dim=-1))
        value = torch.arccos(vec[:, i - 1] / temp)
        if i != vec.shape[1] - 1:
            thetas[:, i] = value
        else:
            mask = vec[:, i] >= 0
            thetas[mask, i] = value[mask]
            thetas[~mask, i] = 2 * math.pi - value[~mask]
    thetas[:, 0] = r
    return thetas


def hyper_sph2cart(coords):
    r, thetas = coords[:, 0], coords[:, 1:]
    vec = torch.zeros_like(coords)
    for i in range(coords.shape[1] - 1):
        if i == 0:
            vec[:, i] = r * torch.cos(thetas[:, i])
        else:
            temp = r * torch.sin(thetas[:, :i]).prod(dim=1)
            vec[:, i] = temp * torch.cos(thetas[:, i])

    vec[:, -1] = r * torch.sin(thetas).prod(dim=-1)
    return vec


def multivariate_normality_torch(X, alpha=0.05):
    from scipy.stats import lognorm
    import time
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps = 1e-8
    # Check input and remove missing values
    X = torch.tensor(X)
    if torch.cuda.is_available():
        X = X.cuda()
    assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)."
    n, p = X.shape
    assert n >= 3, "X must have at least 3 rows."
    assert p >= 2, "X must have at least two columns."

    time_series = [time.time()]
    # Covariance matrix
    mean = torch.mean(X, dim=0, keepdim=True)
    centered = X - mean
    S = (centered.t() @ centered) / (centered.shape[0] - 1 + eps)

    S_inv = torch.pinverse(S).to(X.dtype)
    S_inv = np.linalg.pinv(S, hermitian=True).astype(X.dtype)  # Preserving original dtype
    difT = centered

    time_series.append(time.time())
    output_time(time_series) # 1

    # Squared-Mahalanobis distances
    Dj = torch.diag(torch.mm(torch.mm(difT, S_inv), difT.T))
    Y = torch.mm(torch.mm(X, S_inv), X.T)
    Djk = -2 * Y.T + torch.repeat_interleave(torch.diag(Y.T), n).view(n, -1) + torch.tile(torch.diag(Y.T), (n, 1))

    time_series.append(time.time())
    output_time(time_series)  # 2

    # Smoothing parameter
    b = 1 / (torch.sqrt(torch.tensor(2., device=device))) * ((2 * p + 1) / 4) ** (1 / (p + 4)) * (n ** (1 / (p + 4)))

    # Is matrix full-rank (columns are linearly independent)?
    if torch.matrix_rank(S) == p:
        hz = n * (
                1 / (n ** 2) * torch.sum(torch.sum(torch.exp(-(b ** 2) / 2 * Djk)))
                - 2 * ((1 + (b ** 2)) ** (-p / 2))
                * (1 / n)
                * (torch.sum(torch.exp(-((b ** 2) / (2 * (1 + (b ** 2)))) * Dj)))
                + ((1 + (2 * (b ** 2))) ** (-p / 2))
        )
    else:
        hz = n * 4

    time_series.append(time.time())
    output_time(time_series)  # 3


    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2
    # Mean and variance
    mu = 1 - a ** (-p / 2) * (1 + p * b**2 / a + (p * (p + 2) * (b**4)) / (2 * a**2))
    si2 = (
        2 * (1 + 4 * b**2) ** (-p / 2)
        + 2
        * a ** (-p)
        * (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4))
        - 4
        * wb ** (-p / 2)
        * (1 + (3 * p * b**4) / (2 * wb) + (p * (p + 2) * b**8) / (2 * wb**2))
    )

    time_series.append(time.time())
    output_time(time_series)  # 4


    # Lognormal mean and variance
    scale = torch.sqrt(mu**4 / (si2 + mu**2))
    psi = torch.sqrt(torch.log1p(si2 / mu**2))

    time_series.append(time.time())
    output_time(time_series)  # 5


    # P-value
    pval = lognorm.sf(hz.cpu().numpy(), psi.cpu().numpy(), scale=scale.cpu().numpy())
    normal = True if pval > alpha else False

    time_series.append(time.time())
    output_time(time_series)  # 6

    return hz, pval, normal



def output_time(ser):
    begin = ser[0]
    for i, t in enumerate(ser):
        if i == 0:
            continue
        print("Idx: {}, time: {:.3f}".format(i, t - begin))
        begin = t


class LoadAndInit(object):
    def __init__(self, data_path):
        self.path = data_path
        self.dim = 256
        self.device = 'cpu'
        self.files = next(os.walk(data_path))[2]
        self.predicate_container_vis = [[] for _ in range(51)]
        self.predicate_container_prd = [[] for _ in range(51)]
        self.triplet_container_vis = dict()
        self.triplet_container_prd = dict()

    def load_and_record(self):
        print("loading pkl .....")
        for name in tqdm(self.files):
            fp = os.path.join(self.path, name)
            fp_data = torch.load(fp)
            p_vis, p_prd = fp_data['p_vis'], fp_data['p_prd']
            t_vis, t_prd = fp_data['t_vis'], fp_data['t_prd']
            label, pair = fp_data['label'], fp_data['pair']
            self.predicate(p_vis, p_prd, label)
            self.triplet(t_vis, t_prd, label, pair)

    def predicate(self, vis, prd, label):
        coord_vis = cart2hyper_sph(vis)
        coord_prd = cart2hyper_sph(prd)
        for i in range(len(label)):
            idx = int(label[i].item())
            self.device = vis[i].device
            self.predicate_container_vis[idx].append(coord_vis[i])
            self.predicate_container_prd[idx].append(coord_prd[i])

    def triplet(self, vis, prd, label, pair):
        coord_vis = cart2hyper_sph(vis)
        coord_prd = cart2hyper_sph(prd)
        for i in range(len(label)):
            p = int(label[i].item())
            s, o = int(pair[i][0].item()), int(pair[i][1].item())
            if (s, o, p) in self.triplet_container_vis:
                self.triplet_container_vis[(s, o, p)].append(coord_vis[i])
                self.triplet_container_prd[(s, o, p)].append(coord_prd[i])
            else:
                self.triplet_container_vis[(s, o, p)] = [coord_vis[i]]
                self.triplet_container_prd[(s, o, p)] = [coord_prd[i]]

    def mean_cov(self):
        predicate_vis_mean, predicate_vis_cov = [], []
        predicate_prd_mean, predicate_prd_cov = [], []
        triplet_vis_mean, triplet_vis_cov = dict(), dict()
        triplet_prd_mean, triplet_prd_cov = dict(), dict()
        print("predicate vis/prd mean/cov calculate.....")
        for i in tqdm(range(len(self.predicate_container_vis))):
            if len(self.predicate_container_vis[i]) == 0:
                predicate_vis_mean.append(torch.zeros((1, self.dim), device=self.device))
                predicate_vis_cov.append(torch.zeros((self.dim, self.dim), device=self.device))
            else:
                mean, cov = self.mean_cov_item(self.predicate_container_vis[i])
                predicate_vis_mean.append(mean.to(self.device))
                predicate_vis_cov.append(cov.to(self.device))

            if len(self.predicate_container_prd[i]) == 0:
                predicate_prd_mean.append(torch.zeros((1, self.dim), device=self.device))
                predicate_prd_cov.append(torch.zeros((self.dim, self.dim), device=self.device))
            else:
                mean, cov = self.mean_cov_item(self.predicate_container_prd[i])
                predicate_prd_mean.append(mean.to(self.device))
                predicate_prd_cov.append(cov.to(self.device))

        predicate_vis_mean = torch.cat(predicate_vis_mean, dim=0)
        predicate_vis_cov = torch.stack(predicate_vis_cov, dim=0)
        predicate_prd_mean = torch.cat(predicate_prd_mean, dim=0)
        predicate_prd_cov = torch.stack(predicate_prd_cov, dim=0)

        print("triplet vis/prd mean/cov calculate.....")
        for key, value in self.triplet_container_vis.items():
            mean, cov = self.mean_cov_item(value)
            triplet_vis_mean[key] = mean
            triplet_vis_cov[key] = cov

        for key, value in self.triplet_container_prd.items():
            mean, cov = self.mean_cov_item(value)
            triplet_prd_mean[key] = mean
            triplet_prd_cov[key] = cov

        result = {
            "predicate_vis": [predicate_vis_mean, predicate_vis_cov],
            "predicate_prd": [predicate_prd_mean, predicate_prd_cov],
            "triplet_vis": [triplet_vis_mean, triplet_vis_cov],
            "triplet_prd": [triplet_prd_mean, triplet_prd_cov],
        }
        dst_path = '/'.join(self.path.split('/')[: -1])
        torch.save(result, os.path.join(dst_path, "mean_cov.pkl"))
        return result

    @staticmethod
    def mean_cov_item(input_list):
        eps = 1e-8
        container = torch.stack(input_list, dim=0)
        if torch.cuda.is_available():
            container = container.cuda()
        mean = torch.mean(container, dim=0, keepdim=True)
        centered = container - mean
        cov_matrix = (centered.t() @ centered) / (centered.shape[0] - 1 + eps)
        cov_matrix = cov_matrix.cpu()
        """
        if float(container.shape[0]) > float(container.shape[1]):
            p_value = multivariate_normality_torch(container.cpu().numpy(), alpha=.05)[1]
            print(p_value)
        """
        return mean, cov_matrix

    def run(self):
        self.load_and_record()
        self.mean_cov()


# TODO: according to the config path, load and compute the multi-valued normal distribution.
path = "/mnt/data/yangruijie/hook/checkpoint/ContrastiveHie/predcls/epoch_feat"
lai = LoadAndInit(path)
lai.run()

