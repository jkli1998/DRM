"""
Based on the implementation of https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import copy
import torch
import torch.nn as nn
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import ScaledDotProductAttention,\
    MultiHeadAttention, PositionwiseFeedForward

class Single_Att_Layer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, att_wo_drop=False):
        super(Single_Att_Layer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, att_wo_drop=att_wo_drop)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q_input, k_input, v_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q_input, k_input, v_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()
        return enc_output, enc_slf_attn

class Self_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, input_feats, num_objs):
        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)
        # -- Prepare masks
        bsz = len(num_objs)
        device = input_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1).clone() # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)
        slf_attn_mask = self.zero_check(num_objs, slf_attn_mask)
        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
                input_feats, input_feats, input_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        enc_output = enc_output[non_pad_mask.squeeze(-1)]

        return enc_output

    @staticmethod
    def zero_check(num_objs, slf_attn_mask):
        non_zero = [i for i, v in enumerate(num_objs) if v != 0]
        for i, v in enumerate(num_objs):
            if v == 0:
                nz_idx = non_zero[0]
                slf_attn_mask[i] = slf_attn_mask[nz_idx].clone()
        return slf_attn_mask

class Cross_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, visual_feats, textual_feats, num_objs):
        visual_feats = visual_feats.split(num_objs, dim=0)
        visual_feats = nn.utils.rnn.pad_sequence(visual_feats, batch_first=True)
        textual_feats = textual_feats.split(num_objs, dim=0)
        textual_feats = nn.utils.rnn.pad_sequence(textual_feats, batch_first=True)
        # -- Prepare masks
        bsz = len(num_objs)
        device = visual_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1).clone() # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)
        slf_attn_mask = self.zero_check(num_objs, slf_attn_mask)
        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
                visual_feats, textual_feats, textual_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output

    @staticmethod
    def zero_check(num_objs, slf_attn_mask):
        non_zero = [i for i, v in enumerate(num_objs) if v != 0]
        for i, v in enumerate(num_objs):
            if v == 0:
                nz_idx = non_zero[0]
                slf_attn_mask[i] = slf_attn_mask[nz_idx].clone()
        return slf_attn_mask


class Entity2PredicateCrossAttentionEncoder(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, att_wo_drop=True)

    def forward(self, predicate_feat, entity_feat, rel_pairs, num_objs, num_edges):
        predicate_feat = predicate_feat.split(num_edges, dim=0)
        predicate_feat = nn.utils.rnn.pad_sequence(predicate_feat, batch_first=True)
        entity_feat = entity_feat.split(num_objs, dim=0)
        entity_feat = nn.utils.rnn.pad_sequence(entity_feat, batch_first=True)

        bsz = len(num_objs)
        device = predicate_feat.device
        pad_q_len = max(num_edges)
        pad_v_len = max(num_objs)
        num_edges_ = torch.LongTensor(num_edges).to(device).unsqueeze(1).expand(-1, pad_q_len)
        non_pad_mask = torch.arange(pad_q_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_edges_).unsqueeze(-1)

        slf_attn_mask = torch.ones((bsz, pad_q_len, pad_v_len), device=device, dtype=torch.bool)
        pos_mask = torch.arange(pad_q_len, device=device).view(1, -1).expand(bsz, -1).ge(num_edges_).unsqueeze(
            1).expand(-1, pad_v_len, -1).transpose(1, 2)
        slf_attn_mask = slf_attn_mask & (~pos_mask)
        for idx, i_rel_pairs in enumerate(rel_pairs):
            rel_ind = torch.arange(len(i_rel_pairs), device=i_rel_pairs.device).unsqueeze(-1)
            slf_attn_mask[idx, rel_ind, i_rel_pairs] = False
        predicate_output, _ = self.transformer_layer(
            predicate_feat, entity_feat, entity_feat,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask)
        predicate_output = predicate_output[non_pad_mask.squeeze(-1)]
        return predicate_output


class Predicate2EntityCrossAttentionEncoder(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, att_wo_drop=True)

    def forward(self, entity_feat, predicate_feat, rel_pairs, num_objs, num_edges):
        predicate_feat = predicate_feat.split(num_edges, dim=0)
        predicate_feat = nn.utils.rnn.pad_sequence(predicate_feat, batch_first=True)
        entity_feat = entity_feat.split(num_objs, dim=0)
        entity_feat = nn.utils.rnn.pad_sequence(entity_feat, batch_first=True)

        bsz = len(num_objs)
        device = predicate_feat.device
        pad_v_len = max(num_edges)
        pad_q_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_q_len)
        non_pad_mask = torch.arange(pad_q_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1)

        slf_attn_mask = torch.ones((bsz, pad_q_len, pad_v_len), device=device, dtype=torch.bool)
        pos_mask = torch.arange(pad_q_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(
            1).expand(-1, pad_v_len, -1).transpose(1, 2)
        slf_attn_mask = slf_attn_mask & (~pos_mask)
        for idx, i_rel_pairs in enumerate(rel_pairs):
            rel_ind = torch.arange(len(i_rel_pairs), device=i_rel_pairs.device).unsqueeze(-1)
            slf_attn_mask[idx, i_rel_pairs, rel_ind] = False
        entity_output, _ = self.transformer_layer(
            entity_feat, predicate_feat, predicate_feat,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask)
        entity_output = entity_output[non_pad_mask.squeeze(-1)]
        return entity_output
