import os
import torch
from tqdm import tqdm
from torch.nn import functional as F

class LoadAndInit(object):
    def __init__(self, data_path, n_class,):
        self.path = data_path
        self.predicate_container_vis = [[] for _ in range(n_class)]
        self.predicate_container_prd = [[] for _ in range(n_class)]
        self.triplet_container_vis = dict()
        self.triplet_container_prd = dict()
        self.predicate_tpt_style_vis = [dict() for _ in range(n_class)]
        self.predicate_tpt_style_prd = [dict() for _ in range(n_class)]
        self.tpt_cnt = dict()

    def load_and_record(self):
        print("loading pkl .....")
        files = next(os.walk(self.path))[2]
        for name in tqdm(files):
            fp = os.path.join(self.path, name)
            fp_data = torch.load(fp, map_location=torch.device('cpu'))
            p_vis, p_prd = fp_data['p_vis'], fp_data['p_prd']
            t_vis, t_prd = fp_data['t_vis'], fp_data['t_prd']
            label, pair = fp_data['label'], fp_data['pair']
            self.predicate(p_vis, p_prd, label)
            self.triplet(t_vis, t_prd, label, pair)
            # self.triplet_style(p_vis, p_prd, label, pair)

    def predicate(self, vis, prd, label):
        for i in range(len(label)):
            idx = int(label[i].item())
            self.predicate_container_vis[idx].append(vis[i])
            self.predicate_container_prd[idx].append(prd[i])

    def triplet_style(self, vis, prd, label, pair):
        for i in range(len(label)):
            p = int(label[i].item())
            s, o = int(pair[i][0].item()), int(pair[i][1].item())
            if (s, o, p) in self.predicate_tpt_style_vis[p]:
                self.predicate_tpt_style_vis[p][(s, o, p)].append(vis[i])
                self.predicate_tpt_style_prd[p][(s, o, p)].append(prd[i])
            else:
                self.predicate_tpt_style_vis[p][(s, o, p)] = [vis[i]]
                self.predicate_tpt_style_prd[p][(s, o, p)] = [prd[i]]

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

    def save_data(self):
        path = "/".join(self.path.split('/')[:-1])
        feats = {
            "p_vis": self.predicate_container_vis,
            "p_prd": self.predicate_container_prd,
            "t_vis": self.triplet_container_vis,
            "t_prd": self.triplet_container_prd,
            # "triplet_style_p_vis": self.predicate_tpt_style_vis,
            # "triplet_style_p_prd": self.predicate_tpt_style_prd,
            "tpt_cnt": self.tpt_cnt,
        }
        torch.save(feats, os.path.join(path, "feat.pkl"))

    def run(self):
        self.load_and_record()
        self.save_data()


def main():
    path = "/irip/guoxiefan_2023/hook/SHA-GCL-for-SGG/checkpoint_new/rebuttal/pc-sgcls2.5/epoch_feat"
    n_class = 101
    lai = LoadAndInit(path, n_class)
    lai.run()


main()

