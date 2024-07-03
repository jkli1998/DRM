import os
from collections import OrderedDict, defaultdict
from typing import Dict
import numpy as np
import pickle
import torch


def median_resampling_dict_generation(dataset, cfg, category_list, logger):
    F_c = np.zeros(len(category_list))
    for i in range(len(dataset.relationships)):
        relation = dataset.relationships[i].copy()
        all_rel_sets = defaultdict(list)
        for (o0, o1, r) in relation:
            all_rel_sets[(o0, o1)].append(r)
        for value in all_rel_sets.values():
            p = 1.0 / len(value)
            for v in value:
                F_c[v] += p
    eps = 1e-11
    rc_cls = {
        i: 1 for i in range(len(category_list))
    }
    prob = np.ones_like(F_c)
    prob = prob / len(prob[1:])

    old_freq = F_c / np.sum(F_c)
    freq = prob / (old_freq + eps)
    freq[0] = 0.0
    media = np.median(freq[1:])
    freq = freq / media
    prob[0] = 0.0
    reverse_fc = freq[1:]
    for i, rc in enumerate(reverse_fc.tolist()):
        rc_cls[i + 1] = rc
    repeat_instance = []
    for i in range(len(dataset.relationships)):
        relation = dataset.relationships[i].copy()
        gt_classes = dataset.gt_classes[i].copy()
        o1o2 = gt_classes[relation[:, :2]]
        for (o1, o2), gtr in zip(o1o2, relation[:, 2]):
            qr = quantization(rc_cls[gtr])
            if qr - 1 > 0:
                for _ in range(qr-1):
                    repeat_instance.append((o1, o2, gtr))
    return repeat_instance, freq, prob


def median_resampling_dict_generation_sgdet(dataset, cfg, category_list, logger):
    feats = torch.load(cfg.MODEL.ROI_RELATION_HEAD.RECONSTRUCT.FILE_PATH, map_location=torch.device('cpu'))
    F_c = np.zeros(len(category_list))
    p_vis = feats['p_vis']
    t_vis = feats['t_vis']

    for i in range(len(F_c)):
        F_c[i] += len(p_vis[i])
    eps = 1e-11
    rc_cls = {
        i: 1 for i in range(len(category_list))
    }
    prob = np.ones_like(F_c)
    prob = prob / len(prob[1:])

    old_freq = F_c / np.sum(F_c)
    freq = prob / (old_freq + eps)
    freq[0] = 0.0
    media = np.median(freq[1:])
    freq = freq / media
    prob[0] = 0.0
    reverse_fc = freq[1:]
    for i, rc in enumerate(reverse_fc.tolist()):
        rc_cls[i + 1] = rc
    repeat_instance = []
    for key, value in t_vis.items():
        gtr = key[-1]
        if rc_cls[gtr] - 1 > 0:
            qr = quantization((rc_cls[gtr] - 1) * len(value))
            for _ in range(qr):
                repeat_instance.append((key[0], key[1], key[2]))
    return repeat_instance, freq, prob


def quantization(float_num):
    _int_part = int(float_num)
    _frac_part = float_num - _int_part
    rands = np.random.rand(1)[0]
    _int_part += (rands < _frac_part).astype(int)
    return _int_part


def down_sample_num(pred_matrix, freq, prob, logger, factor):
    prob[0] = 0.0
    est = np.median(pred_matrix[1:])
    down_rate = 1.0 - (pred_matrix - est) / (pred_matrix + 1e-10)
    down_rate[0] = 0.0
    down_rate[down_rate < 1.0] = down_rate[down_rate < 1.0] ** factor
    generate_rate = (freq - 1) / (freq + 1e-10)
    generate_rate[generate_rate < 0.0] = 0.0
    generate_rate[0] = 0.0
    # logger.info("down_rate: " + " ".join([str(x) for x in down_rate.tolist()]))
    # logger.info("down_rate: " + " ".join([str(x) for x in down_rate.tolist()]))
    # logger.info("generate_rate: " + " ".join([str(x) for x in generate_rate.tolist()]))
    # logger.info("calibrate_list all: " + str(est))
    return down_rate, generate_rate

