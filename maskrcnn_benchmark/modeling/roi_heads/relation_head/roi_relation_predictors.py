# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import copy
import time
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
import logging
from maskrcnn_benchmark.modeling.utils import cat
from .model_motifs import LSTMContext, FrequencyBias
from .model_two_stream import TwoStreamContext
from maskrcnn_benchmark.data import get_dataset_statistics

import random
from .utils_sample import BalanceTripletSample
from .reconstruction import PredicateReconstruction, TripletReconstruction

def make_roi_relation_predictor(cfg, in_channels):
    import time
    result_str = '---'*20
    result_str += ('\n\nthe dataset we use is [ %s ]' % cfg.GLOBAL_SETTING.DATASET_CHOICE)
    result_str += ('\nthe model we use is [ %s ]' % cfg.GLOBAL_SETTING.RELATION_PREDICTOR)
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ predcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ sgcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == False:
        result_str += ('\ntraining mode is [ sgdet ]')
    else:
        exit('wrong training mode!')
    result_str += ('\nlearning rate is [ %.5f ]' % cfg.SOLVER.BASE_LR)
    result_str += '\n\n'
    result_str += '---'*20
    print(result_str)
    time.sleep(2)
    func = registry.ROI_RELATION_PREDICTOR[cfg.GLOBAL_SETTING.RELATION_PREDICTOR]
    return func(cfg, in_channels)


def filter_zero_graph(proposals, rel_pair_idxs):
    filtered_proposals = []
    filtered_pair_idxs = []
    half = len(proposals) // 2
    half_cnt = 0
    for idx, prop in enumerate(proposals):
        if len(prop) != 0:
            filtered_proposals.append(prop)
            filtered_pair_idxs.append(rel_pair_idxs[idx])
            if idx < half:
                half_cnt += 1
    return filtered_proposals, half_cnt, filtered_pair_idxs


def filter_empty_bbox(proposals, rel_pair_idxs, rel_labels, roi_features, union_features):
    result_dict = {
        "aug_masks": [],
        "rel_masks": [],
        "entity_maps": [],
        "filtered_proposals": [],
        "filtered_rel_pairs": [],
        "filtered_rel_labels": [],
        "filtered_roi_features": None,
        "filtered_union_features": None,
    }
    for i, (i_prop, i_rel_pair) in enumerate(zip(proposals, rel_pair_idxs)):
        i_aug_mask = i_prop.get_field("aug_mask").bool()
        # aug-mask check none
        result_dict["aug_masks"].append(i_aug_mask)
        result_dict["filtered_proposals"].append(i_prop[i_aug_mask])
        filter_nums = torch.nonzero(~i_aug_mask).squeeze(-1)
        filter_mask_s = i_rel_pair[:, 0][:, None] == filter_nums[None, :]
        filter_mask_o = i_rel_pair[:, 1][:, None] == filter_nums[None, :]
        i_filter_mask = ~(filter_mask_s | filter_mask_o).any(dim=1)
        # filter invalid i_rel_pair
        result_dict["rel_masks"].append(i_filter_mask)
        if rel_labels is not None:
            i_rel_labels = rel_labels[i]
            filtered_label = i_rel_labels[i_filter_mask]
            result_dict["filtered_rel_labels"].append(filtered_label)
        else:
            result_dict["filtered_rel_labels"] = None
        filtered_pairs = i_rel_pair[i_filter_mask, :]
        rel_filter_num = torch.nonzero(i_aug_mask).squeeze(-1)
        rel_range = torch.arange(len(rel_filter_num), dtype=rel_filter_num.dtype, device=rel_filter_num.device)
        i_mapping = {k.item(): v.item() for k, v in zip(rel_filter_num, rel_range)}
        i_mapped0 = torch.tensor([i_mapping[x.item()] for x in filtered_pairs[:, 0]], device=i_rel_pair.device, dtype=torch.int64)
        i_mapped1 = torch.tensor([i_mapping[x.item()] for x in filtered_pairs[:, 1]], device=i_rel_pair.device, dtype=torch.int64)
        i_mapped_rel_pairs = torch.stack((i_mapped0, i_mapped1), dim=1)
        result_dict["entity_maps"].append(i_mapping)
        result_dict["filtered_rel_pairs"].append(i_mapped_rel_pairs)
    all_aug_masks = torch.cat(result_dict["aug_masks"], dim=0)
    all_rel_masks = torch.cat(result_dict["rel_masks"], dim=0)
    result_dict["filtered_roi_features"] = roi_features[all_aug_masks]
    result_dict["filtered_union_features"] = union_features[all_rel_masks]
    return result_dict


@registry.ROI_RELATION_PREDICTOR.register("DRMPredictor")
class DRMPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(DRMPredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        # load class dict
        statistics = get_dataset_statistics(config)
        self.statistics = statistics
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)
        # module construct
        self.context_layer = TwoStreamContext(config, obj_classes, rel_classes, in_channels)

        self.contrast_dim = config.MODEL.ROI_RELATION_HEAD.RECONSTRUCT.CONTRAST_DIM
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.register_buffer("cnt", torch.zeros(1)[0])
        if self.config.MODEL.STAGE == "stage2":
            self.balance_sample = BalanceTripletSample(config, statistics)

        # predicate weights
        assert self.config.MODEL.ROI_RELATION_HEAD.UNION_COMPRESS
        self.compress_dim = config.MODEL.ROI_RELATION_HEAD.UNION_COMPRESS_DIM # TODO: change
        self.predicate_post_cat = nn.Linear(self.hidden_dim, self.compress_dim)
        self.predicate_vis_proj = nn.Sequential(
            nn.Linear(self.compress_dim, self.compress_dim),
            nn.ReLU(),
            nn.Linear(self.compress_dim, self.contrast_dim),
        )
        self.predicate_prd_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.contrast_dim),
        )
        # triplet weights
        self.triplet_post_cat = nn.Linear(self.hidden_dim, self.compress_dim)
        self.triplet_vis_proj = nn.Sequential(
            nn.Linear(self.compress_dim, self.compress_dim),
            nn.ReLU(),
            nn.Linear(self.compress_dim, self.contrast_dim),
        )
        self.triplet_prd_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.contrast_dim),
        )

        self.reset_tag = False
        input_dim = self.compress_dim + self.hidden_dim
        self.triplet_predictor = nn.Sequential(
            nn.Linear(input_dim, self.compress_dim),
            nn.ReLU(),
            nn.Linear(self.compress_dim, self.num_rel_cls)
        )
        self.predicate_predictor = nn.Sequential(
            nn.Linear(input_dim, self.compress_dim),
            nn.ReLU(),
            nn.Linear(self.compress_dim, self.num_rel_cls)
        )

        self.fg_bg_loss = nn.BCEWithLogitsLoss()
        self.criterion_loss = nn.CrossEntropyLoss()
        self.logger = logging.getLogger("maskrcnn_benchmark.roi_relation_predictor")
        import random
        if self.config.MODEL.STAGE == "stage2":
            self.repeat_dict = self.statistics['repeat_dict']
            random.shuffle(self.repeat_dict)
            self.ave_gen = 1.0 * len(self.repeat_dict) / statistics[
                'data_length'] * config.SOLVER.IMS_PER_BATCH / config.TEST.IMS_PER_BATCH
            self.repeat_pointer = 0
            data = self.load_data_src()
            self.predicate_construct = PredicateReconstruction(config, self.num_rel_cls, statistics, data)
            self.triplet_construct = TripletReconstruction(config, self.hidden_dim, self.compress_dim, self.num_rel_cls, statistics, data)
    
    def load_data_src(self):
        from .reconstruction import mean_cov, zero_split, LoadAndInit
        path = os.path.join(self.config.OUTPUT_DIR, 'infer_train_feat')
        feats = LoadAndInit(path, self.num_rel_cls).run()
        self.logger.info("\nend loading feats")
        so_dim, vis_dim = self.hidden_dim, self.compress_dim
        truncate = np.min([x[-1] for x in self.statistics['repeat_dict']])
        data = mean_cov(feats, so_dim, vis_dim, 16, truncate)
        return data

    def reset_parameters(self):
        if self.reset_tag:
            return
        if self.config.MODEL.STAGE == "stage2":
            self.triplet_predictor[0].reset_parameters()
            self.triplet_predictor[2].reset_parameters()
            self.predicate_predictor[0].reset_parameters()
            self.predicate_predictor[2].reset_parameters()
        self.reset_tag = True

    def one_epoch(self, share_pack, label, idx):
        p_vis, p_prd, pair = share_pack["p_vis"], share_pack["p_prd"], share_pack["pair_gt"]
        t_vis, t_prd = share_pack["t_vis"], share_pack["t_prd"]
        self.cnt += 1
        fg_mask = label != 0
        flip_path = "infer_train_feat"
        path = os.path.join(self.config.OUTPUT_DIR, flip_path)
        os.makedirs(path, exist_ok=True)
        data = {
            'idx': idx,
            'p_vis': p_vis[fg_mask],
            'p_prd': p_prd[fg_mask],
            't_vis': t_vis[fg_mask],
            't_prd': t_prd[fg_mask],
            'label': label[fg_mask],
            'pair': pair[fg_mask],
        }
        torch.save(data, os.path.join(path, "{}.pkl".format(self.cnt)))

    @staticmethod
    def quantization(float_num):
        _int_part = int(float_num)
        _frac_part = float_num - _int_part
        rands = np.random.rand(1)[0]
        _int_part += (rands < _frac_part).astype(int)
        return _int_part

    def generate_from_repeat(self, gen_num):
        gen_relation = self.repeat_dict[self.repeat_pointer: self.repeat_pointer + gen_num]
        self.repeat_pointer = self.repeat_pointer + gen_num
        if self.repeat_pointer >= len(self.repeat_dict):
            self.repeat_pointer = 0
            random.shuffle(self.repeat_dict)
        # gen_relation
        p_prd, p_vis, p_label = self.predicate_construct.generate(gen_relation)
        t_prd, t_vis, _ = self.triplet_construct.generate(gen_relation)
        pair_pred = torch.tensor([[x[0], x[1]] for x in gen_relation], dtype=torch.long, device=self.config.MODEL.DEVICE)
        return p_prd, p_vis, t_prd, t_vis, p_label, pair_pred

    def share_forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, union_tpt):
        chosen_matrix = chosen_ind = None
        cond_use = self.training and self.config.MODEL.STAGE == "stage1"
        cond_use_box = self.config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
        cond_pre_com = self.config.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX
        if cond_use and (cond_use_box or cond_pre_com):
            result_dict = filter_empty_bbox(proposals, rel_pair_idxs, rel_labels, roi_features, union_features)
            roi_features = result_dict["filtered_roi_features"]
            union_features = result_dict["filtered_union_features"]
            union_features_p = union_features_t = union_features

            proposals, rel_pair_idxs = result_dict["filtered_proposals"], result_dict["filtered_rel_pairs"]
            rel_labels = result_dict["filtered_rel_labels"]
            filtered_zero_proposals, _, filtered_pairs = filter_zero_graph(result_dict["filtered_proposals"], rel_pair_idxs)
            obj_labels = cat([proposal.get_field("labels") for proposal in filtered_zero_proposals], dim=0)
        else:
            filtered_zero_proposals = proposals
            filtered_pairs = rel_pair_idxs
            obj_labels = cat([proposal.get_field("labels") for proposal in filtered_zero_proposals], dim=0)
            union_features_p = union_features
            union_features_t = union_tpt
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, filtered_zero_proposals, filtered_pairs,
                                                            union_features_p, union_features_t)
        edge_ctx_prd, edge_ctx_tpt = edge_ctx
        p_vis, p_prd = self.encoder_predicate(edge_ctx_prd, union_features)
        t_vis, t_prd = self.encoder_triplet(edge_ctx_tpt, union_features)

        obj_preds = obj_preds.split(num_objs, dim=0)

        gt_pair = None
        if self.training or self.config.MODEL.INFER_TRAIN:
            fg_labels = [proposal.get_field("labels") for proposal in proposals]
            gt_pair = []
            for pair_idx, obj_lbl in zip(rel_pair_idxs, fg_labels):
                gt_pair.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))
            gt_pair = cat(gt_pair, dim=0)

        pair_pred = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
            pair_pred.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        pair_pred = cat(pair_pred, dim=0)

        predicate_vis, predicate_prd = self.decoder_predicate(p_vis, p_prd)
        triplet_vis, triplet_prd = self.decoder_triplet(t_vis, t_prd)

        result = {
            "p_vis": p_vis,
            "p_prd": p_prd,
            "t_vis": t_vis,
            "t_prd": t_prd,
            "cpr_p_vis": predicate_vis,
            "cpr_p_prd": predicate_prd,
            "cpr_t_vis": triplet_vis,
            "cpr_t_prd": triplet_prd,
            "chosen_ind": chosen_ind,
            "pair_pred": pair_pred,
            "pair_gt": gt_pair,
            "obj_dists": obj_dists,
            "num_objs": num_objs,
            "num_rels": num_rels,
            "r_label": rel_labels,
            "o_label": obj_labels,
        }
        return result

    def encoder_predicate(self, predicate_feat, union_features):
        ctx_gate = self.predicate_post_cat(predicate_feat)
        visual_rep = ctx_gate * union_features
        return visual_rep, predicate_feat

    def encoder_triplet(self, triplet_feat, union_features):
        ctx_gate = self.triplet_post_cat(triplet_feat)
        visual_rep = ctx_gate * union_features
        return visual_rep, triplet_feat

    def decoder_predicate(self, visual_rep, prod_rep):
        visual_rep = self.predicate_vis_proj(visual_rep)
        prod_rep = self.predicate_prd_proj(prod_rep)
        visual_rep = F.normalize(visual_rep, p=2, dim=-1)
        prod_rep = F.normalize(prod_rep, p=2, dim=-1)
        return visual_rep, prod_rep

    def decoder_triplet(self, visual_rep, prod_rep):
        visual_rep = self.triplet_vis_proj(visual_rep)
        prod_rep = self.triplet_prd_proj(prod_rep)
        visual_rep = F.normalize(visual_rep, p=2, dim=-1)
        prod_rep = F.normalize(prod_rep, p=2, dim=-1)
        return visual_rep, prod_rep

    @staticmethod
    def split_v1_v2(x, num_rels):
        half_num = len(num_rels) // 2
        sp_x = x.split(num_rels, dim=0)
        v1_x = sp_x[: half_num]
        v2_x = sp_x[half_num: ]
        return torch.cat(v1_x, dim=0), torch.cat(v2_x, dim=0)

    @staticmethod
    def fg_mask(vis, prd, pair, label, chosen_ind):
        if chosen_ind is not None:
            label = label[chosen_ind]
            assert (label != 0).all()
            vis = vis[chosen_ind]
            prd = prd[chosen_ind]
            pair = pair[chosen_ind]
        else:
            fg_mask = label != 0
            vis, prd, pair, label = vis[fg_mask], prd[fg_mask], pair[fg_mask], label[fg_mask]
        tpt = torch.cat((pair, label.view(-1, 1)), dim=-1)
        return vis, prd, tpt, label

    @staticmethod
    def chosen_mask(vis, prd, pair, label, chosen_ind):
        if chosen_ind is not None:
            label = label[chosen_ind]
            assert (label != 0).all()
            vis = vis[chosen_ind]
            prd = prd[chosen_ind]
            pair = pair[chosen_ind]
        tpt = torch.cat((pair, label.view(-1, 1)), dim=-1)
        return vis, prd, tpt, label

    @staticmethod
    def min_wo_zero(input_tensor):
        defined_inf = 10000.0
        input_tensor = input_tensor.detach()
        input_tensor[input_tensor == 0] = defined_inf
        result = torch.min(input_tensor, dim=-1)[0].reshape(-1, 1)
        return result

    def predicate_contrastive_loss(self, v1_vis, v1_prd, v1_tpt, v2_vis, v2_prd, v2_tpt):
        v1_predicate = v1_tpt[:, -1].reshape(-1, 1)
        v2_predicate = v2_tpt[:, -1].reshape(-1, 1)
        union_predicate = torch.cat((v1_predicate, v2_predicate), dim=0)
        predicate_mask = torch.eq(union_predicate, union_predicate.T).float()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(predicate_mask), 1,
                                    torch.arange(len(predicate_mask)).view(-1, 1).to(predicate_mask.device), 0)
        predicate_mask = predicate_mask * logits_mask
        pos_mask = torch.any(predicate_mask.bool(), dim=-1)
        pos_mask = pos_mask & (union_predicate.squeeze(-1) != 0)

        p_vis_loss, p_vis_ind = self.i_contrastive(v1_vis, v2_vis, predicate_mask, pos_mask, tau=0.2)
        p_prd_loss, p_prd_ind = self.i_contrastive(v1_prd, v2_prd, predicate_mask, pos_mask, tau=0.2)
        if p_vis_loss is not None:
            p_vis_loss += p_prd_loss
        if p_vis_ind is None:
            return dict()
        loss = {
            "predicate_contrastive": p_vis_loss
        }
        return loss


    def triplet_contrastive_loss(self, v1_vis, v1_prd, v1_tpt, v2_vis, v2_prd, v2_tpt):
        eps = 1e-6
        v1_predicate = v1_tpt[:, -1].reshape(-1, 1)
        v2_predicate = v2_tpt[:, -1].reshape(-1, 1)
        union_predicate = torch.cat((v1_predicate, v2_predicate), dim=0)
        predicate_mask = torch.eq(union_predicate, union_predicate.T).float()
        union_tpt = torch.cat((v1_tpt, v2_tpt), dim=0)
        triplet_mask = (union_tpt[:, None, :] == union_tpt[None, :, :]).all(-1).float()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(predicate_mask), 1,
                                    torch.arange(len(predicate_mask)).view(-1, 1).to(predicate_mask.device), 0)
        predicate_mask = predicate_mask * logits_mask
        triplet_mask = triplet_mask * logits_mask
        # predicate_mask = torch.eq(predicate, m_predicate.T).float()
        # triplet_mask = (tpt[:, None, :] == m_tpt[None, :, :]).all(-1).float()
        pos_mask = torch.any(triplet_mask.bool(), dim=-1)
        pos_mask = pos_mask & (union_predicate.squeeze(-1) != 0)

        t_vis_loss, t_vis_ind = self.i_contrastive(v1_vis, v2_vis, triplet_mask, pos_mask, tau=0.1)
        t_prd_loss, t_prd_ind = self.i_contrastive(v1_prd, v2_prd, triplet_mask, pos_mask, tau=0.1)
        if t_vis_loss is not None:
            t_vis_loss += t_prd_loss
        if t_vis_ind is None:
            return dict()
        loss = {
            "triplet_contrastive": t_vis_loss
        }
        return loss

    def i_contrastive(self, v1_src, v2_src, lbl_mask, pos_mask=None, tau=0.07):
        # src: [N, C], dst: [M, C], logits_src2dst: [N, M]
        union_src = torch.cat((v1_src, v2_src), dim=0)
        union_dst = union_src
        if pos_mask is not None:
            union_src = union_src[pos_mask]
            lbl_mask = lbl_mask[pos_mask]
        if len(union_src) == 0:
            return None, None

        logits_src2dst = torch.div(torch.matmul(union_src, union_dst.T), tau)
        logits_max, _ = torch.max(logits_src2dst, dim=-1, keepdim=True)
        logits = logits_src2dst - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
        loss = - torch.sum(lbl_mask * log_prob, dim=-1) / torch.sum(lbl_mask, dim=-1)
        # mask indicator
        loss_item = - (lbl_mask * log_prob)
        return loss.mean(), loss_item

    def split_mask(self, vis, prd, pair_gt, cat_labels, chosen_ind, num_rels):
        v1_vis, v2_vis = self.split_v1_v2(vis, num_rels)
        v1_prd, v2_prd = self.split_v1_v2(prd, num_rels)
        v1_pair_gt, v2_pair_gt = self.split_v1_v2(pair_gt, num_rels)
        v1_label, v2_label = self.split_v1_v2(cat_labels, num_rels)
        cid1 = None if chosen_ind is None else chosen_ind[0]
        cid2 = None if chosen_ind is None else chosen_ind[1]
        v1_vis, v1_prd, v1_tpt, v1_label = self.chosen_mask(v1_vis, v1_prd, v1_pair_gt, v1_label, cid1)
        v2_vis, v2_prd, v2_tpt, v2_label = self.chosen_mask(v2_vis, v2_prd, v2_pair_gt, v2_label, cid2)
        return (v1_vis, v1_prd, v1_tpt, v1_label), (v2_vis, v2_prd, v2_tpt, v2_label)

    def prd_tpt_pretrain(self, share_pack, cat_labels):
        loss_dict = dict()
        p_vis, p_prd = share_pack['cpr_p_vis'], share_pack['cpr_p_prd']
        t_vis, t_prd = share_pack['cpr_t_vis'], share_pack['cpr_t_prd']
        pair_pred, pair_gt = share_pack['pair_pred'], share_pack['pair_gt']
        chosen_ind = share_pack['chosen_ind']
        num_rels = share_pack['num_rels']
        pv1_result, pv2_result = self.split_mask(p_vis, p_prd, pair_gt, cat_labels, chosen_ind, num_rels)
        pv1_vis, pv1_prd, pv1_tpt, pv1_label = pv1_result
        pv2_vis, pv2_prd, pv2_tpt, pv2_label = pv2_result
        result = self.predicate_contrastive_loss(pv1_vis, pv1_prd, pv1_tpt, pv2_vis, pv2_prd, pv2_tpt)
        loss_dict.update(result)

        tv1_result, tv2_result = self.split_mask(t_vis, t_prd, pair_gt, cat_labels, chosen_ind, num_rels)
        tv1_vis, tv1_prd, tv1_tpt, tv1_label = tv1_result
        tv2_vis, tv2_prd, tv2_tpt, tv2_label = tv2_result
        result = self.triplet_contrastive_loss(tv1_vis, tv1_prd, tv1_tpt, tv2_vis, tv2_prd, tv2_tpt)
        loss_dict.update(result)
        return loss_dict

    def pretrain_forward(self, share_pack, cat_labels):
        loss_dict = dict()
        loss_pt = self.prd_tpt_pretrain(share_pack, cat_labels)
        loss_dict.update(loss_pt)
        return loss_dict

    def predicate_triplet_clf(self, aug_p_vis, aug_p_prd, aug_t_vis, aug_t_prd, cat_labels, pair_pred, chosen_matrix):
        if chosen_matrix is not None:
            csn_p_vis, csn_p_prd = aug_p_vis[chosen_matrix], aug_p_prd[chosen_matrix]
            csn_t_vis, csn_t_prd = aug_t_vis[chosen_matrix], aug_t_prd[chosen_matrix]
            csn_cat_labels, csn_pair_pred = cat_labels[chosen_matrix], pair_pred[chosen_matrix]
        else:
            csn_p_vis, csn_p_prd = aug_p_vis, aug_p_prd
            csn_t_vis, csn_t_prd = aug_t_vis, aug_t_prd
            csn_cat_labels, csn_pair_pred = cat_labels, pair_pred
        loss_dict = dict()
        if csn_cat_labels.shape[0] > 0:
            cat_feat_p = torch.cat((csn_p_vis, csn_p_prd), dim=-1)
            cat_feat_t = torch.cat((csn_t_vis, csn_t_prd), dim=-1)
            rel_dists = self.predicate_predictor(cat_feat_p) + self.triplet_predictor(cat_feat_t)
            if self.use_bias:
                cls_bias = self.freq_bias.index_with_labels(csn_pair_pred.long())
                rel_dists += cls_bias
            loss_dict['rel_loss'] = self.criterion_loss(rel_dists, csn_cat_labels) * 36
        return loss_dict

    @staticmethod
    def reduce_ratio_then_sample(rel_labels, aug_rate):
        assert not (aug_rate > 1.0).any()
        fg_aug_rate = aug_rate[rel_labels]
        rands = torch.rand(len(rel_labels)).to(rel_labels.device)
        aug_mask = rands < fg_aug_rate
        return aug_mask

    def finetune_forward(self, share_pack, cat_labels):
        chosen_matrix = None
        if self.config.MODEL.STAGE == "stage2":
            chosen_matrix = self.balance_sample.generate_cur_chosen_matrix(cat_labels)
        losses = self.finetune_prd_tpt(share_pack, cat_labels, chosen_matrix)
        return losses

    def finetune_prd_tpt(self, share_pack, cat_labels, chosen_matrix):
        pair_pred = share_pack["pair_gt"]
        if self.config.MODEL.STAGE == "stage2":
            p_vis, p_prd = share_pack["p_vis"].detach(), share_pack["p_prd"].detach()
            t_vis, t_prd = share_pack["t_vis"].detach(), share_pack["t_prd"].detach()
        else:
            p_vis, p_prd = share_pack["p_vis"], share_pack["p_prd"]
            t_vis, t_prd = share_pack["t_vis"], share_pack["t_prd"]

        if chosen_matrix is not None:
            p_vis, p_prd = p_vis[chosen_matrix], p_prd[chosen_matrix]
            t_vis, t_prd = t_vis[chosen_matrix], t_prd[chosen_matrix]
            cat_labels, pair_pred = cat_labels[chosen_matrix], pair_pred[chosen_matrix]
        if self.config.MODEL.STAGE == "stage2":
            gen_num = self.quantization(self.ave_gen)
            if gen_num > 0:
                aug_p_prd, aug_p_vis, aug_t_prd, aug_t_vis, p_label, aug_pair_pred = self.generate_from_repeat(gen_num)
                p_prd = torch.cat((p_prd, aug_p_prd), dim=0)
                p_vis = torch.cat((p_vis, aug_p_vis), dim=0)
                t_prd = torch.cat((t_prd, aug_t_prd), dim=0)
                t_vis = torch.cat((t_vis, aug_t_vis), dim=0)
                cat_labels = torch.cat((cat_labels, torch.cat(p_label, dim=0)), dim=0)
                pair_pred = torch.cat((pair_pred, aug_pair_pred), dim=0)
        losses = self.predicate_triplet_clf(
            p_vis, p_prd, t_vis, t_prd, cat_labels, pair_pred, None
        )
        return losses

    def test_predicate(self, share_pack):
        pair_pred = share_pack["pair_pred"]
        p_vis, p_prd = share_pack["p_vis"], share_pack["p_prd"]
        p_rel_bias = None
        if self.use_bias:
            p_rel_bias = self.freq_bias.index_with_labels(pair_pred.long())
        p_compact = cat((p_vis, p_prd), dim=-1)
        p_rel_dists = self.predicate_predictor(p_compact)
        if p_rel_bias is not None:
            p_rel_dists += p_rel_bias
        p_rel_prob = torch.softmax(p_rel_dists, dim=-1)
        return p_rel_prob

    def test_triplet(self, share_pack):
        pair_pred = share_pack["pair_pred"]
        t_vis, t_prd = share_pack["t_vis"], share_pack["t_prd"]
        t_rel_bias = None
        if self.use_bias:
            t_rel_bias = self.freq_bias.index_with_labels(pair_pred.long())
        t_compact = cat((t_vis, t_prd), dim=-1)
        t_rel_dists = self.triplet_predictor(t_compact)
        if t_rel_bias is not None:
            t_rel_dists += t_rel_bias
        t_rel_prob = torch.softmax(t_rel_dists, dim=-1)
        return t_rel_prob

    def test_prd_tpt(self, share_pack):
        pair_pred = share_pack["pair_pred"]
        p_vis, p_prd = share_pack["p_vis"], share_pack["p_prd"]
        t_vis, t_prd = share_pack["t_vis"], share_pack["t_prd"]
        p_compact = cat((p_vis, p_prd), dim=-1)
        t_compact = cat((t_vis, t_prd), dim=-1)
        p_rel_logits = self.predicate_predictor(p_compact)
        t_rel_logits = self.triplet_predictor(t_compact)
        rel_logits = p_rel_logits + t_rel_logits
        if self.use_bias:
            rel_logits += self.freq_bias.index_with_labels(pair_pred.long())
        rel_prob = torch.softmax(rel_logits, dim=-1)
        return rel_prob

    def test_forward(self, share_pack):
        rel_dists = self.test_prd_tpt(share_pack)
        obj_dists = share_pack['obj_dists']
        num_objs = share_pack["num_objs"]
        obj_dists = obj_dists.split(num_objs, dim=0)
        num_rels = share_pack["num_rels"]
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, union_features_tpt):
        if self.config.MODEL.STAGE == "stage1":
            share_pack = self.share_forward(
                proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, union_features_tpt
            )
        else:
            with torch.no_grad():
                share_pack = self.share_forward(
                    proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, union_features_tpt
                )
        add_losses = dict()
        if self.training:
            rel_labels = share_pack['r_label']
            cat_labels = cat(rel_labels, dim=0)
            if self.config.MODEL.STAGE == "stage1":
                result = self.pretrain_forward(share_pack, cat_labels)
                loss_ft = self.finetune_forward(share_pack, cat_labels)
                add_losses.update(result)
                add_losses.update(loss_ft)
            else:
                self.reset_parameters()
                add_losses = self.finetune_forward(share_pack, cat_labels)
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL and self.config.MODEL.STAGE == "stage1":
                obj_dists = share_pack['obj_dists']
                fg_labels = share_pack['o_label']
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj * 5
            return None, None, add_losses
        else:
            if self.config.MODEL.INFER_TRAIN:
                cat_labels = cat(rel_labels, dim=0)
                idx = [proposal.get_field("image_index") for proposal in proposals]
                self.one_epoch(share_pack, cat_labels, idx)
            obj_dists, rel_dists = self.test_forward(share_pack)
            return obj_dists, rel_dists, add_losses




