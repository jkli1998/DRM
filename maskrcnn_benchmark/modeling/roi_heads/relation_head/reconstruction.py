import os
import copy
import torch
import logging
import torch.nn as nn
import numpy as np
from tqdm import tqdm
eps = 1e-3


class ConstructBase(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def i_predicate_sim(self, ctt_mean, q_mean):
        dist = torch.pairwise_distance(q_mean.unsqueeze(0), ctt_mean)
        sim = - dist * 2
        return sim


    def i_predicate_construct(self, ctt_mean, ctt_cov, sim):
        weight = torch.softmax(sim, dim=-1)
        re_mean = torch.sum(weight[:, None] * ctt_mean, dim=0)
        re_cov = torch.sum(weight[:, None, None] * ctt_cov.float(), dim=0)
        return re_mean, re_cov

    def i_triplet_sim(self, ctt_mean, q_mean):
        dist = torch.pairwise_distance(q_mean.unsqueeze(0), ctt_mean)
        sim = - dist * 2
        return sim

    def i_triplet_construct_mean(self, ctt_mean, ctt_cov, sim):
        v, ind = torch.topk(sim, k=100)
        weight = torch.softmax(v, dim=-1)
        re_mean = torch.sum(weight[:, None] * ctt_mean[ind], dim=0)
        re_cov = torch.sum(weight[:, None, None] * ctt_cov[ind].float(), dim=0)
        return re_mean, re_cov


class Head2TailPredicate(ConstructBase):
    def __init__(self, cfg, start_cls, num_class, rate):
        super(Head2TailPredicate, self).__init__(cfg)
        self.start_cls = start_cls
        self.num_class = num_class
        self.rate = rate.cpu()

    def predicate_reconstruct(self, vis_mean, vis_cov, prd_mean, prd_cov):
        re_vis_mean = copy.deepcopy(vis_mean)
        re_vis_cov = copy.deepcopy(vis_cov)
        re_prd_mean = copy.deepcopy(prd_mean)
        re_prd_cov = copy.deepcopy(prd_cov)

        for i in tqdm(range(self.start_cls, self.num_class)):
            q_vis = copy.deepcopy(vis_mean[i])
            q_prd = copy.deepcopy(prd_mean[i])
            sim_vis = self.i_predicate_sim(vis_mean[: self.start_cls], q_vis)
            sim_prd = self.i_predicate_sim(prd_mean[: self.start_cls], q_prd)
            sim1 = sim_vis
            sim2 = sim_prd
            result_vis = self.i_predicate_construct(vis_mean[: self.start_cls], vis_cov[: self.start_cls], sim1)
            result_prd = self.i_predicate_construct(prd_mean[: self.start_cls], prd_cov[: self.start_cls], sim2)
            alpha = self.rate[i]
            re_vis_cov[i] = result_vis[1] * alpha + vis_cov[i] * (1 - alpha)
            re_prd_cov[i] = result_prd[1] * alpha + prd_cov[i] * (1 - alpha)
        return re_vis_mean, re_vis_cov, re_prd_mean, re_prd_cov


class Head2TailTriplet(ConstructBase):
    def __init__(self, cfg, rate, idx_mapper):
        super(Head2TailTriplet, self).__init__(cfg)
        self.tpt_rate = torch.zeros(len(idx_mapper))
        rate = rate.cpu()
        for tpt, idx in idx_mapper.items():
            rel_label = tpt[-1]
            self.tpt_rate[idx] = rate[rel_label]

    def triplet_reconstruct(self, query, support):
        query_mean_prd, query_mean_vis, query_cov_prd, query_cov_vis = query
        support_mean_prd, support_mean_vis, support_cov_prd, support_cov_vis = support
        re_vis_mean = copy.deepcopy(query_mean_vis)
        re_prd_mean = copy.deepcopy(query_mean_prd)
        re_vis_cov = copy.deepcopy(query_cov_vis)
        re_prd_cov = copy.deepcopy(query_cov_prd)
        for idx in tqdm(range(len(query_mean_vis))):
            q_vis = copy.deepcopy(query_mean_vis[idx])
            q_prd = copy.deepcopy(query_mean_prd[idx])
            sim_vis = self.i_triplet_sim(support_mean_vis, q_vis)
            sim_prd = self.i_triplet_sim(support_mean_prd, q_prd)
            sim1 = sim_vis
            sim2 = sim_prd
            result_vis = self.i_triplet_construct_mean(support_mean_vis, support_cov_vis, sim1)
            result_prd = self.i_triplet_construct_mean(support_mean_prd, support_cov_prd, sim2)
            alpha = self.tpt_rate[idx]
            re_vis_cov[idx] = result_vis[1] * alpha + query_cov_vis[idx] * (1 - alpha)
            re_prd_cov[idx] = result_prd[1] * alpha + query_cov_prd[idx] * (1 - alpha)
        return re_vis_mean, re_vis_cov, re_prd_mean, re_prd_cov


class PredicateReconstruction(nn.Module):
    def __init__(self, cfg, num_classes, statistics, data):
        super(PredicateReconstruction, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.statistics = statistics
        self.mean_vis = self.cov_vis = None
        self.mean_prd = self.cov_prd = None
        # three type reconstruction from head 2 tail / group (intra/inter).
        self.generate_rate = torch.from_numpy(self.statistics['generate_rate']).to(cfg.MODEL.DEVICE)
        self.start_cls = zero_split(self.generate_rate)
        self.load_and_init(data)

    def load_and_init(self, data):
        mean_vis, cov_vis = data['predicate_vis']
        mean_prd, cov_prd = data['predicate_prd']
        predicate = Head2TailPredicate(self.cfg, self.start_cls, self.num_classes, self.generate_rate)
        result = predicate.predicate_reconstruct(mean_vis, cov_vis, mean_prd, cov_prd)
        self.mean_vis, self.cov_vis, self.mean_prd, self.cov_prd = result

    def get_sample_ratio(self):
        return self.generate_rate

    @torch.no_grad()
    def generate(self, gen_relations):
        labels = torch.tensor([x[-1] for x in gen_relations], dtype=torch.long, device=self.cfg.MODEL.DEVICE)
        aug_prd_feats, aug_vis_feats, aug_label = self.self_aug_generate(labels)
        if len(aug_prd_feats) > 0:
            aug_prd_feats = torch.cat(aug_prd_feats, dim=0)
            aug_vis_feats = torch.cat(aug_vis_feats, dim=0)
        return aug_prd_feats, aug_vis_feats, aug_label

    def self_aug_generate(self, fg_labels):
        if len(fg_labels) == 0:
            return [], [], []
        aug_prd_feats, aug_vis_feats, aug_label = [], [], []
        uni_fg_labels = torch.unique(fg_labels, dim=0)
        for i in range(len(uni_fg_labels)):
            lbl = int(uni_fg_labels[i].item())
            num = int(torch.sum(fg_labels == lbl).item())
            aug_prd = self.generate_from_mean_cov(
                self.mean_prd[lbl], self.cov_prd[lbl].float(), num
            ).to(self.cfg.MODEL.DEVICE)
            aug_prd_feats.append(aug_prd)
            aug_vis = self.generate_from_mean_cov(
                self.mean_vis[lbl], self.cov_vis[lbl].float(), num
            ).to(self.cfg.MODEL.DEVICE)
            aug_vis_feats.append(aug_vis)
            aug_label.append(uni_fg_labels[i].repeat(num))
        return aug_prd_feats, aug_vis_feats, aug_label

    @staticmethod
    def generate_from_mean_cov(mean, cov, num):
        if not torch.linalg.cholesky_ex(cov).info.eq(0).unsqueeze(0):
            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device)
        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        samples = m.sample((num,))
        # samples = mean.unsqueeze(0).repeat(num ,1)
        return samples


class TripletReconstruction(nn.Module):
    def __init__(self, cfg, so_shape, vis_shape, num_classes, statistics, data):
        super(TripletReconstruction, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.statistics = statistics
        self.prd_shape = so_shape
        self.vis_shape = vis_shape
        self.gen_rate = torch.from_numpy(self.statistics['generate_rate']).to(cfg.MODEL.DEVICE)
        support_num = 64

        self.re_utils = {
            "start_cls": zero_split(self.gen_rate),
            "idx_mapper": dict(),
            "supp_idx_mapper": dict(),
        }
        statistics['fg_matrix'][:, :, 0] = 0
        statistics['fg_matrix'][:, 0, :] = 0
        statistics['fg_matrix'][0, :, :] = 0
        idx_cnt, supp_idx_cnt = 0, 0
        indexes = torch.nonzero(statistics['fg_matrix'])
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and self.cfg.GLOBAL_SETTING.DATASET_CHOICE != "GQA_200":
            for idx in tqdm(range(len(indexes))):
                s, o, r = indexes[idx, 0].item(), indexes[idx, 1].item(), indexes[idx, 2].item()
                num = statistics['fg_matrix'][s, o, r].item()
                if num == 0:
                    continue
                if r < self.re_utils['start_cls']:
                    if num >= support_num:
                        self.re_utils['supp_idx_mapper'][(s, o, r)] = supp_idx_cnt
                        supp_idx_cnt += 1
                else:
                    self.re_utils['idx_mapper'][(s, o, r)] = idx_cnt
                    idx_cnt += 1
        else:
            for key, num in data['tpt_cnt'].items():
                s, o, r = key
                if r < self.re_utils['start_cls']:
                    if num >= support_num:
                        self.re_utils['supp_idx_mapper'][(s, o, r)] = supp_idx_cnt
                        supp_idx_cnt += 1
                else:
                    self.re_utils['idx_mapper'][(s, o, r)] = idx_cnt
                    idx_cnt += 1
        self.mean_vis = self.mean_prd = None
        self.cov_vis = self.cov_prd = None
        self.logger = logging.getLogger("maskrcnn_benchmark.reconstruction")
        self.tpt_reconstruct = Head2TailTriplet(self.cfg, self.gen_rate, self.re_utils['idx_mapper'])
        self.load_and_init_src(data)

    @staticmethod
    def cov_item(input_list):
        eps = 1e-8
        container = torch.stack(input_list, dim=0)
        mean = torch.mean(container, dim=0, keepdim=True)
        centered = container - mean
        cov_matrix = (centered.t() @ centered) / (centered.shape[0] - 1 + eps)
        cov_matrix = cov_matrix
        return cov_matrix

    def load_and_init_src(self, data):
        tpt_cnt = data['tpt_cnt']
        mean_vis, cov_vis = data['triplet_vis']
        mean_prd, cov_prd = data['triplet_prd']
        query, support = self.memory_mapper_src(mean_vis, cov_vis, mean_prd, cov_prd)
        self.logger.info("begin triplet reconstruct...query: {}, support: {}".format(len(query[0]), len(support[0])))
        result = self.tpt_reconstruct.triplet_reconstruct(query, support)
        self.mean_vis, self.cov_vis, self.mean_prd, self.cov_prd = result

    def memory_mapper_src(self, mean_vis, cov_vis, mean_prd, cov_prd):
        query_mean_prd = torch.zeros((len(self.re_utils['idx_mapper']), self.prd_shape))
        query_mean_vis = torch.zeros((len(self.re_utils['idx_mapper']), self.vis_shape))
        query_cov_prd = torch.zeros((len(self.re_utils['idx_mapper']), self.prd_shape, self.prd_shape))
        query_cov_vis = torch.zeros((len(self.re_utils['idx_mapper']), self.vis_shape, self.vis_shape))
        support_mean_prd = torch.zeros((len(self.re_utils['supp_idx_mapper']), self.prd_shape))
        support_mean_vis = torch.zeros((len(self.re_utils['supp_idx_mapper']), self.vis_shape))
        support_cov_prd = torch.zeros((len(self.re_utils['supp_idx_mapper']), self.prd_shape, self.prd_shape))
        support_cov_vis = torch.zeros((len(self.re_utils['supp_idx_mapper']), self.vis_shape, self.vis_shape))
        for tpt_key, tpt_idx in tqdm(self.re_utils['idx_mapper'].items()):
            if tpt_key in mean_vis.keys():
                query_mean_vis[tpt_idx] = mean_vis[tpt_key]
                query_mean_prd[tpt_idx] = mean_prd[tpt_key]
                if tpt_key in cov_vis.keys():
                    query_cov_vis[tpt_idx] = cov_vis[tpt_key]
                    query_cov_prd[tpt_idx] = cov_prd[tpt_key]
                else:
                    self.logger.info("in mean-vis not in cov,record query miss: {}, cnt: {}".format(
                        tpt_idx, self.statistics['fg_matrix'][tpt_key[0], tpt_key[1], tpt_key[2]].item())
                    )
            else:
                self.logger.info("query miss: {}, cnt: {}".format(
                    tpt_idx, self.statistics['fg_matrix'][tpt_key[0], tpt_key[1], tpt_key[2]].item()
                ))
        for tpt_key, tpt_idx in tqdm(self.re_utils['supp_idx_mapper'].items()):
            if tpt_key in mean_vis.keys():
                support_mean_vis[tpt_idx] = mean_vis[tpt_key]
                support_mean_prd[tpt_idx] = mean_prd[tpt_key]
                support_cov_vis[tpt_idx] = cov_vis[tpt_key]
                support_cov_prd[tpt_idx] = cov_prd[tpt_key]
            else:
                self.logger.info("support miss: {}, cnt: {}".format(tpt_idx, self.statistics['fg_matrix'][tpt_key[0], tpt_key[1], tpt_key[2]].item()))
        query = [query_mean_prd, query_mean_vis, query_cov_prd, query_cov_vis]
        support = [support_mean_prd, support_mean_vis, support_cov_prd, support_cov_vis]
        return query, support

    def get_sample_ratio(self):
        return self.gen_rate

    @torch.no_grad()
    def generate(self, gen_relation):
        gen_relation = torch.tensor(gen_relation, dtype=torch.long, device=self.cfg.MODEL.DEVICE)
        aug_prd_feats, aug_vis_feats, aug_label = self.self_aug_construct(gen_relation[:, :2], gen_relation[:, 2])
        aug_prd_feats = torch.cat(aug_prd_feats, dim=0)
        aug_vis_feats = torch.cat(aug_vis_feats, dim=0)
        return aug_prd_feats, aug_vis_feats, aug_label

    @staticmethod
    def generate_from_mean_cov(mean, cov, num):
        if not torch.linalg.cholesky_ex(cov).info.eq(0).unsqueeze(0):
            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device)
        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        samples = m.sample((num,))
        # samples = mean.unsqueeze(0).repeat(num ,1)
        return samples

    """
    def generate_from_mean_cov(self, mean, cov, num, src_mean):
        if not torch.linalg.cholesky_ex(cov).info.eq(0).unsqueeze(0):
            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device)
        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        samples = []
        if self.cfg.MODEL.ROI_RELATION_HEAD.RECONSTRUCT.GENERATE_COMPARE_SRC:
            assert src_mean is not None
            dst_mean = src_mean[None, :]
        else:
            dst_mean = mean[None, :]
        cnt = 0
        while cnt != num:
            item = m.sample((1,))
            item_vec = item
            sim = torch.mm(item_vec, dst_mean.t())[0, 0].item()
            if sim > self.cfg.MODEL.ROI_RELATION_HEAD.RECONSTRUCT.GENERATE_TPT_SIM_THR:
                samples.append(item)
                cnt += 1
        samples = torch.cat(samples, dim=0)
        return samples
    """

    def self_aug_construct(self, pairs, fg_labels):
        aug_prd_feats, aug_vis_feats, aug_label = [], [], []
        tpt = torch.cat((pairs, fg_labels.unsqueeze(-1)), dim=-1)
        uni_tpt = torch.unique(tpt, dim=0)
        for i in range(len(uni_tpt)):
            r = int(uni_tpt[i, 2].item())
            s, o = int(uni_tpt[i, 0].item()), int(uni_tpt[i, 1].item())
            mask = (uni_tpt[i][None, :] == tpt).all(1)
            num = int(torch.sum(mask).item())
            if (s, o, r) not in self.re_utils['idx_mapper']:
                self.logger.info("_".join([str(s), str(o), str(r)]) + " miss in dix mapper")
                continue
            idx = self.re_utils['idx_mapper'][(s, o, r)]
            aug_vis = self.generate_from_mean_cov(
                self.mean_vis[idx], self.cov_vis[idx], num
            ).to(self.cfg.MODEL.DEVICE)
            aug_vis_feats.append(aug_vis)
            aug_prd = self.generate_from_mean_cov(
                self.mean_prd[idx], self.cov_prd[idx], num
            ).to(self.cfg.MODEL.DEVICE)
            aug_prd_feats.append(aug_prd)
            aug_label.append(uni_tpt[i, 2].repeat(num))
        return aug_prd_feats, aug_vis_feats, aug_label


def mean_cov(feats, so_dim, vis_dim, min_num, truncate):
    predicate_container_vis = feats['p_vis']
    predicate_container_prd = feats['p_prd']
    triplet_container_vis = feats['t_vis']
    triplet_container_prd = feats['t_prd']
    tpt_cnt = feats['tpt_cnt']

    predicate_vis_mean, predicate_vis_cov = [], []
    predicate_prd_mean, predicate_prd_cov = [], []
    triplet_vis_mean, triplet_vis_cov = dict(), dict()
    triplet_prd_mean, triplet_prd_cov = dict(), dict()
    print("predicate vis/prd mean/cov calculate.....")
    for i in tqdm(range(len(predicate_container_vis))):
        if len(predicate_container_vis[i]) == 0:
            predicate_vis_mean.append(torch.zeros((1, vis_dim)))
            predicate_vis_cov.append(torch.zeros((vis_dim, vis_dim)))

            predicate_prd_mean.append(torch.zeros((1, so_dim), ))
            predicate_prd_cov.append(torch.zeros((so_dim, so_dim), ))
        else:
            mean, cov = mean_cov_item(predicate_container_vis[i])
            predicate_vis_mean.append(mean.cpu())
            predicate_vis_cov.append(cov.cpu())
            mean, cov = mean_cov_item(predicate_container_prd[i])
            predicate_prd_mean.append(mean.cpu())
            predicate_prd_cov.append(cov.cpu())
    predicate_vis_mean = torch.cat(predicate_vis_mean, dim=0)
    predicate_vis_cov = torch.stack(predicate_vis_cov, dim=0)
    predicate_prd_mean = torch.cat(predicate_prd_mean, dim=0)
    predicate_prd_cov = torch.stack(predicate_prd_cov, dim=0)

    print("triplet vis/prd mean/cov calculate.....")
    print("triplet length. ", len(triplet_container_vis))
    for key, value in triplet_container_vis.items():
        mean, cov = mean_cov_item(value)
        triplet_vis_mean[key] = mean.cpu()
        if len(value) >= min_num:
            triplet_vis_cov[key] = cov.cpu()
        else:
            if key[-1] >= truncate:
                triplet_vis_cov[key] = cov.cpu()

    for key, value in triplet_container_prd.items():
        mean, cov = mean_cov_item(value)
        triplet_prd_mean[key] = mean.cpu()
        if len(value) >= min_num:
            triplet_prd_cov[key] = cov.cpu()
        else:
            if key[-1] >= truncate:
                triplet_prd_cov[key] = cov.cpu()
    print("saved triplet length. ", len(triplet_vis_mean))
    print("saved triplet length cov. ", len(triplet_vis_cov))
    result = {
        "predicate_vis": [predicate_vis_mean, predicate_vis_cov],
        "predicate_prd": [predicate_prd_mean, predicate_prd_cov],
        "triplet_vis": [triplet_vis_mean, triplet_vis_cov],
        "triplet_prd": [triplet_prd_mean, triplet_prd_cov],
        "tpt_cnt": tpt_cnt,
    }
    return result

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

def quantization(rel_repeat_list):
    # scale by alpha before input.
    rands = torch.rand(*rel_repeat_list.shape).to(rel_repeat_list.device)
    _int_part = rel_repeat_list.int()
    _frac_part = rel_repeat_list - _int_part
    rep_factors = _int_part + (rands < _frac_part).int()
    return rep_factors


def cls2generate(fg_labels, ratio, class_number=51):
    # fg_labels numpy
    # return cls with aug number
    fg_ratio = ratio[fg_labels]
    fg_repeat = quantization(fg_ratio)
    """
    gen_num = torch.zeros(class_number, device=fg_labels.device)
    for i in range(class_number):
        mask = fg_labels == i
        if not mask.any():
            continue
        gen_num[i] = torch.sum(fg_repeat[mask])
    """
    return fg_repeat


def scale_factor(update_times, alpha_array, cut_off=64):
    """resist the less data with unstable alpha"""
    if len(update_times.shape) == 0:
        update_times = update_times.unsqueeze(0)
        alpha_array = alpha_array.unsqueeze(0)
    mask = update_times > cut_off
    up = copy.deepcopy(update_times)
    up[up > cut_off] = cut_off
    scale = 1 - torch.pow(1 - alpha_array, up)
    scale[mask] = 1.0
    return scale


def zero_split(ratio):
    for i, item in enumerate(ratio):
        if item > 0.0:
            return i
    return len(ratio) - 1


class LoadAndInit(object):
    def __init__(self, data_path, n_class,):
        self.path = data_path
        self.predicate_container_vis = [[] for _ in range(n_class)]
        self.predicate_container_prd = [[] for _ in range(n_class)]
        self.triplet_container_vis = dict()
        self.triplet_container_prd = dict()
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

    def save_data(self):
        path = "/".join(self.path.split('/')[:-1])
        feats = {
            "p_vis": self.predicate_container_vis,
            "p_prd": self.predicate_container_prd,
            "t_vis": self.triplet_container_vis,
            "t_prd": self.triplet_container_prd,
            "tpt_cnt": self.tpt_cnt,
        }
        # torch.save(feats, os.path.join(path, "feat.pkl"))
        return feats

    def run(self):
        self.load_and_record()
        return self.save_data()


