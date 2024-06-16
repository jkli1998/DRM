import os
import torch
from tqdm import tqdm
from torch.nn import functional as F

class LoadAndInit(object):
    def __init__(self, data_path, vis_dim, so_dim, min_num, n_class, one_epoch, truncate, n_times):
        self.path = data_path
        self.vis_dim = vis_dim
        self.so_dim = so_dim
        self.min_num = min_num
        self.predicate_container_vis = [[] for _ in range(n_class)]
        self.predicate_container_prd = [[] for _ in range(n_class)]
        self.triplet_container_vis = dict()
        self.triplet_container_prd = dict()
        self.tpt_cnt = dict()
        self.one_epoch = one_epoch
        self.truncate = truncate
        self.n_times = n_times

    def n_times_name(self):
        files = next(os.walk(self.path))[2]
        # idx = [int(x.split('.')[0]) for x in files]
        # re_f = []
        # for i, f in zip(idx, files):
        #     if i <= self.one_epoch * self.n_times:
        #         re_f.append(f)
        # return re_f
        return files

    def load_and_record(self):
        print("loading pkl .....")
        files = self.n_times_name()
        for name in tqdm(files):
            fp = os.path.join(self.path, name)
            fp_data = torch.load(fp, map_location=torch.device('cpu'))
            p_vis, p_prd = fp_data['p_vis'], fp_data['p_prd']
            t_vis, t_prd = fp_data['t_vis'], fp_data['t_prd']
            label, pair = fp_data['label'], fp_data['pair']
            self.predicate(p_vis, p_prd, label)
            self.triplet(t_vis, t_prd, label, pair)

    def predicate(self, vis, prd, label):
        for i in range(len(label)):
            idx = int(label[i].item())
            self.predicate_container_vis[idx].append(vis[i])
            self.predicate_container_prd[idx].append(prd[i])

    def triplet(self, vis, prd, label, pair):
        for i in range(len(label)):
            p = int(label[i].item())
            s, o = int(pair[i][0].item()), int(pair[i][1].item())
            if (s, o, p) in self.triplet_container_vis:
                self.triplet_container_vis[(s, o, p)].append(vis[i])
                self.triplet_container_prd[(s, o, p)].append(prd[i])
                self.tpt_cnt[(s, o, p)] += 1
            else:
                self.triplet_container_vis[(s, o, p)] = [vis[i]]
                self.triplet_container_prd[(s, o, p)] = [prd[i]]
                self.tpt_cnt[(s, o, p)] = 1

    def mean_cov(self):
        predicate_vis_mean, predicate_vis_cov = [], []
        predicate_prd_mean, predicate_prd_cov = [], []
        triplet_vis_mean, triplet_vis_cov = dict(), dict()
        triplet_prd_mean, triplet_prd_cov = dict(), dict()
        triplet_vis_cov_low, triplet_prd_cov_low = dict(), dict()
        print("predicate vis/prd mean/cov calculate.....")
        for i in tqdm(range(len(self.predicate_container_vis))):
            if len(self.predicate_container_vis[i]) == 0:
                predicate_vis_mean.append(torch.zeros((1, self.vis_dim)))
                predicate_vis_cov.append(torch.zeros((self.vis_dim, self.vis_dim)))

                predicate_prd_mean.append(torch.zeros((1, self.so_dim), ))
                predicate_prd_cov.append(torch.zeros((self.so_dim, self.so_dim), ))
            else:
                mean, cov = self.mean_cov_item(self.predicate_container_vis[i])
                predicate_vis_mean.append(mean)
                predicate_vis_cov.append(cov)
                mean, cov = self.mean_cov_item(self.predicate_container_prd[i])
                predicate_prd_mean.append(mean)
                predicate_prd_cov.append(cov)

        predicate_vis_mean = torch.cat(predicate_vis_mean, dim=0)
        predicate_vis_cov = torch.stack(predicate_vis_cov, dim=0)
        predicate_prd_mean = torch.cat(predicate_prd_mean, dim=0)
        predicate_prd_cov = torch.stack(predicate_prd_cov, dim=0)

        print("triplet vis/prd mean/cov calculate.....")
        print("triplet length. ", len(self.triplet_container_vis))
        for key, value in self.triplet_container_vis.items():
            mean, cov = self.mean_cov_item(value)
            triplet_vis_mean[key] = mean
            if len(value) >= self.min_num:
                triplet_vis_cov[key] = cov
            else:
                if key[-1] >= self.truncate:
                    triplet_vis_cov_low[key] = value

        for key, value in self.triplet_container_prd.items():
            mean, cov = self.mean_cov_item(value)
            triplet_prd_mean[key] = mean
            if len(value) >= self.min_num:
                triplet_prd_cov[key] = cov
            else:
                if key[-1] >= self.truncate:
                    triplet_prd_cov_low[key] = value
        print("saved triplet length. ", len(triplet_vis_mean))
        print("saved triplet length cov. ", len(triplet_vis_cov))
        result = {
            "predicate_vis": [predicate_vis_mean, predicate_vis_cov],
            "predicate_prd": [predicate_prd_mean, predicate_prd_cov],
            "triplet_vis": [triplet_vis_mean, triplet_vis_cov, triplet_vis_cov_low],
            "triplet_prd": [triplet_prd_mean, triplet_prd_cov, triplet_prd_cov_low],
            "tpt_cnt": self.tpt_cnt,
        }
        dst_path = '/'.join(self.path.split('/')[: -1])
        torch.save(result, os.path.join(dst_path, "mean_cov_{}.pkl".format(self.n_times)))
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


def main():
    one_epoch = 3608
    path = "/irip/guoxiefan_2023/hook/SHA-GCL-for-SGG/checkpoint_new/TwoStream11_36_wospunion_woprdlang_0201/predcls2/epoch_feat_one"
    n_class = 51
    # 小于truncate的都不记录
    truncate = (n_class - 1) // 2
    support_num = 16
    # for n_times in (1, 2, 4, 8, 16):
    if True:
        n_times = 1
        dimension_vis = 1024
        dimension_so = 512
        min_tpt_number = support_num * n_times
        lai = LoadAndInit(path, dimension_vis, dimension_so, min_tpt_number, n_class, one_epoch, truncate, n_times)
        lai.run()


main()
