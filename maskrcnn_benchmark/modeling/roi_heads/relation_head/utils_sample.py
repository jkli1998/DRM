import torch
import random
import numpy as np

"""
class BalanceTripletSample(object):
    def __init__(self, statistics):
        self.sample_rate_list = self.generate_sample(statistics)

    @staticmethod
    def generate_sample(statistics):
        predicate_order_count = statistics['pred_matrix']
        assert len(predicate_order_count) == 51
        median = np.median(predicate_order_count)
        out = [0 for _ in range(len(predicate_order_count))]
        for i in range(len(predicate_order_count)):
            if i == 0:
                out[i] = 0.0
                continue
            if predicate_order_count[i] > median:
                num = 1.0 * median / predicate_order_count[i]
                if num < 0.01:
                    num = 0.01
                out[i] = num
            else:
                out[i] = 1.0
        return out

    def generate_cur_chosen_matrix(self, rel_labels):
        cur_chosen_matrix = torch.zeros_like(rel_labels, dtype=torch.bool)
        for i in range(len(rel_labels)):
            rel_tar = rel_labels[i].item()
            if rel_tar == 0:
                cur_chosen_matrix[i] = False
            else:
                if random.random() <= self.sample_rate_list[rel_tar]:
                    cur_chosen_matrix[i] = True
        return cur_chosen_matrix
"""


class BalanceTripletSample(object):
    def __init__(self, cfg, statistics):
        self.cfg = cfg
        self.sample_rate_list = self.generate_sample(statistics)

    def generate_sample(self, statistics):
        down_rate = statistics['down_rate'].tolist()
        for i in range(len(down_rate)):
            if down_rate[i] < 0.01:
                down_rate[i] = 0.01
        down_rate[0] = 0.0
        return down_rate

    def generate_cur_chosen_matrix(self, rel_labels):
        cur_chosen_matrix = []
        for i in range(len(rel_labels)):
            rel_tar = rel_labels[i].item()
            if rel_tar != 0:
                if random.random() <= self.sample_rate_list[rel_tar]:
                    cur_chosen_matrix.append(i)
            else:
                cur_chosen_matrix.append(i)
        return cur_chosen_matrix

