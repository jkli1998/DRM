import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_co_attention import Self_Attention_Encoder, Cross_Attention_Encoder
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,\
    to_onehot, nms_overlaps, encode_box_info
from .model_Hybrid_Attention import SHA_Encoder, PredicateEntityEncoder


class TwoStreamContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed_prd = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed_tpt = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed_prd.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed_tpt.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.lin_edge_visual_predicate = nn.Linear(self.hidden_dim + self.in_channels, 2 * self.hidden_dim)
        self.lin_edge_visual_obj = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        self.compress_dim = config.MODEL.ROI_RELATION_HEAD.UNION_COMPRESS_DIM
        self.prd_init_fusion = nn.Linear(self.compress_dim, self.hidden_dim)
        # self.lin_edge_textual_prd = nn.Linear(self.embed_dim, 2 * self.hidden_dim)

        self.lin_edge_visual_tpt = nn.Linear(self.hidden_dim + self.in_channels, 2 * self.hidden_dim)
        self.lin_edge_textual_tpt = nn.Linear(self.embed_dim, 2 * self.hidden_dim)
        self.tpt_init_fusion_vis = nn.Linear(2 * self.hidden_dim + self.compress_dim, self.hidden_dim)
        self.tpt_init_fusion_txt = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        self.context_obj = SHA_Encoder(config, self.obj_layer)
        self.context_edge_prd = PredicateEntityEncoder(config, self.edge_layer)
        self.context_edge_tpt = SHA_Encoder(config, self.edge_layer)

    def forward(self, roi_features, proposals, rel_pair_idxs, union_prd, union_tpt, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer

        num_objs = [len(p) for p in proposals]
        obj_pre_rep_vis = cat((roi_features, pos_embed), -1)
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis)
        obj_pre_rep_txt = obj_embed
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt)
        obj_feats_vis, _, = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs)
        obj_feats = obj_feats_vis

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)

        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        num_edges = [pairs.shape[0] for pairs in rel_pair_idxs]

        # prd context
        edge_ctx_prd = self.edge_prd(roi_features, obj_feats, obj_preds, rel_pair_idxs, union_prd, num_objs, num_edges)
        # tpt context
        edge_ctx_tpt = self.edge_tpt(roi_features, obj_feats, obj_preds, rel_pair_idxs, union_tpt, num_objs, num_edges)

        return obj_dists, obj_preds, (edge_ctx_prd, edge_ctx_tpt)

    def edge_prd(self, roi_features, obj_feats, obj_preds, rel_pair_idxs, union_prd, num_objs, num_rels):
        edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
        edge_pre_rep_txt = self.obj_embed_prd(obj_preds)
        predicate = self.prd_init_fusion(union_prd)
        entity = self.lin_edge_visual_obj(edge_pre_rep_vis)

        predicate, _ = self.context_edge_prd(predicate, entity, rel_pair_idxs, num_objs, num_rels)
        return predicate

    def edge_tpt(self, roi_features, obj_feats, obj_preds, rel_pair_idxs, union_tpt, num_objs, num_tpt):
        edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
        edge_pre_rep_txt = self.obj_embed_tpt(obj_preds)
        edge_pre_rep_vis = self.lin_edge_visual_tpt(edge_pre_rep_vis)
        sbj_vis, obj_vis = self.compose_rep(edge_pre_rep_vis, rel_pair_idxs, num_objs)
        edge_pre_rep_txt = self.lin_edge_textual_tpt(edge_pre_rep_txt)
        sbj_txt, obj_txt = self.compose_rep(edge_pre_rep_txt, rel_pair_idxs, num_objs)
        vis = torch.cat((sbj_vis, obj_vis, union_tpt), dim=-1)
        txt = torch.cat((sbj_txt, obj_txt), dim=-1)
        vis = self.tpt_init_fusion_vis(vis)
        txt = self.tpt_init_fusion_txt(txt)
        edge_ctx_vis, _ = self.context_edge_tpt(vis, txt, num_tpt)
        return edge_ctx_vis

    def compose_rep(self, obj_rep, rel_pair_idxs, num_objs):
        obj_rep = obj_rep.view(obj_rep.size(0), 2, self.hidden_dim)
        head_rep = obj_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = obj_rep[:, 1].contiguous().view(-1, self.hidden_dim)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        sbj_reps, obj_reps = [], []
        for pair_idx, head_rep, tail_rep in zip(rel_pair_idxs, head_reps, tail_reps):
            sbj_reps.append(head_rep[pair_idx[:, 0]])
            obj_reps.append(tail_rep[pair_idx[:, 1]])
        sbj_reps = cat(sbj_reps, dim=0)
        obj_reps = cat(obj_reps, dim=0)
        return sbj_reps, obj_reps

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


