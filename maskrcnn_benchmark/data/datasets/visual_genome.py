import os
import json
import logging
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import pickle
from torch.nn import functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from .sample import median_resampling_dict_generation, down_sample_num, median_resampling_dict_generation_sgdet
from maskrcnn_benchmark.utils.comm import get_rank, synchronize
from maskrcnn_benchmark.extra.group_chosen_function import predicate_new_order, predicate_new_order_name
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, split_boxlist
from maskrcnn_benchmark.modeling.roi_heads.box_head.sampling import make_roi_box_samp_processor

BOX_SCALE = 1024  # Scale at which we have the boxes


class VGDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000, zs_file=None,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False,
                 custom_path=''):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        assert split in {'train', 'val', 'test'}
        self.cfg = cfg
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.zs_file = zs_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.num_im = cfg.DATASETS.SUB_SET

        self.ind_to_classes, self.ind_to_predicates = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        self.ind_to_predicates = predicate_new_order_name
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        filter_empty_rels = filter_empty_rels or self.split == 'train' or self.split == 'val'
        self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.split, self.num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap,
        )

        self.filenames, self.img_info = load_image_filenames(img_dir, image_file)  # length equals to split_mask
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

        logger = logging.getLogger("maskrcnn_benchmark.dataset")
        self.logger = logger

        if self.split == 'train':
            self.relationships = self.filter_relations(self.relationships)

        self.idx_list = list(range(len(self.filenames)))
        self.repeat_dict = None
        self.check_ratio()
        if cfg.MODEL.STAGE == "stage2" and self.split == 'train':
            # creat repeat dict in main process, other process just wait and load
            if get_rank() == 0:
                if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    repeat_result = median_resampling_dict_generation(self, self.cfg, self.ind_to_predicates, logger)
                else:
                    repeat_result = median_resampling_dict_generation_sgdet(self, self.cfg, self.ind_to_predicates, logger)
                self.repeat_dict = repeat_result[0]
                self.freq = repeat_result[1]
                self.prob = repeat_result[2]
                with open(os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl"), "wb") as f:
                    pickle.dump(repeat_result, f)

            synchronize()
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                repeat_result = median_resampling_dict_generation(self, self.cfg, self.ind_to_predicates, logger)
            else:
                repeat_result = median_resampling_dict_generation_sgdet(self, self.cfg, self.ind_to_predicates, logger)
            self.repeat_dict = repeat_result[0]
            self.freq = repeat_result[1]
            self.prob = repeat_result[2]

            # duplicate_idx_list = []
            # for idx in range(len(self.filenames)):
            #     r_c = self.repeat_dict[idx]
            #     duplicate_idx_list.extend([idx for _ in range(r_c)])
            # self.idx_list = duplicate_idx_list
        # self.samp_processor = make_roi_box_samp_processor(cfg)

    def check_ratio(self):
        relation_all = 0
        fg_all = 0
        for i, i_rel in enumerate(self.relationships):
            gt_classes = self.gt_classes[i].copy()
            n_class = len(gt_classes)
            relation_all += n_class * (n_class - 1)
            fg_all += len(i_rel)
        self.logger.info("All potential relationships: {}, Labeled foreground count: {}".format(relation_all, fg_all))
        self.logger.info("ratio of foreground to all potential relationships: {}".format(1.0 * fg_all / relation_all))
        self.logger.info("ratio of foreground to all unlabeled pairs: {}".format(1.0 * fg_all / (relation_all - fg_all)))

    def __getitem__(self, index):
        if self.repeat_dict is not None:
            index = self.idx_list[index]
        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)
        target = self.get_groundtruth(index, False)
        fp_name = self.filenames[index].split('/')[-1]
        targets_len = len(target)
        target.add_field("fp_name", fp_name)
        target.add_field("src_w", img.size[0])
        target.add_field("src_h", img.size[1])
        pre_compute_box = tgt_record = pre_comp_record = pre_comp_result = None
        if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and (self.split == 'train'):
            pre_comp_result = torch.load(os.path.join(self.cfg.DATASETS.VG_BBOX_DIR, fp_name))
        if pre_comp_result is not None:
            boxes_arr = torch.as_tensor(pre_comp_result['bbox']).reshape(-1, 4)
            pre_compute_box = BoxList(boxes_arr, img.size, mode='xyxy')
            tgt_record = target.remove_all_fields()
            target = cat_boxlist([target, pre_compute_box])
            pre_comp_record = {
                'pred_scores': pre_comp_result['pred_scores'],
                'pred_labels': pre_comp_result['pred_labels'],
                'predict_logits': pre_comp_result['predict_logits'],
                'labels': pre_comp_result['labels'],
            }
        if self.split == 'train' and (self.cfg.MODEL.INFER_TRAIN or self.cfg.DATASETS.INFER_BBOX):
            img, target = self.transforms(img, target)
            if pre_comp_record is not None:
                target = self.split_target(target, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
        elif self.split == 'train' and self.cfg.MODEL.STAGE == "stage1":
            img1, target1 = self.transforms(img, target)
            img2, target2 = self.transforms(img, target)
            if pre_comp_record is not None:
                target1 = self.split_target(target1, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
                target2 = self.split_target(target2, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
            return [img1, img2], [target1, target2], index
        elif self.transforms is not None:
            img, target = self.transforms(img, target)
            if pre_comp_record is not None:
                target = self.split_target(target, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
        
        return img, target, index

    @staticmethod
    def split_target(all_boxes, targets_len, pre_compute_len, tgt_record, pre_comp_record):
        resized_boxes = split_boxlist(all_boxes, (targets_len, targets_len + pre_compute_len))
        target = resized_boxes[0]
        pre_compute_box = resized_boxes[1]
        target.add_all_fields(tgt_record[0], tgt_record[1])
        pre_compute_box.add_field("pred_scores", pre_comp_record['pred_scores'])
        pre_compute_box.add_field("pred_labels", pre_comp_record['pred_labels'])
        pre_compute_box.add_field("predict_logits", pre_comp_record['predict_logits'])
        pre_compute_box.add_field("labels", pre_comp_record['labels'])
        target = (target, pre_compute_box)
        return target

    def clip_check(self, src_relations):
        """warning use this after filter relation duplicate"""
        new_relation = []

        for index in range(len(src_relations)):
            img_info = self.get_img_info(index)
            w, h = img_info['width'], img_info['height']
            # important: recover original box from BOX_SCALE
            box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
            box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

            target = BoxList(box, (w, h), 'xyxy')  # xyxy
            relation = self.relationships[index].copy()  # (num_rel, 3)
            # add relation to target
            num_box = len(target)
            relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
            for i in range(relation.shape[0]):
                if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                    if (random.random() > 0.5):
                        relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                else:
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            target.add_field("relation", relation_map, is_triplet=True)
            target.add_field("image_index", index)
            new_target = target.clip_to_image(remove_empty=True)
            if len(new_target) == len(target):
                new_relation.append(src_relations[index])
            else:
                img_rel = new_target.get_field("relation").numpy()
                i_rel = np.nonzero(img_rel)
                new_add = []
                for x in range(len(i_rel[0])):
                    s, o = i_rel[0][x], i_rel[1][x]
                    r = img_rel[s, o]
                    new_add.append([s, o, r])
                if len(new_add) == 0:
                    new_relation.append(np.zeros((0, 3)).astype(np.int32))
                else:
                    new_relation.append(np.array(new_add))
        return new_relation

    @staticmethod
    def filter_relations(relations):
        new_relation = []
        for i in range(len(relations)):
            relation = relations[i].copy()
            all_rel_sets = defaultdict(list)
            new_relation_item = []
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            for k, v in all_rel_sets.items():
                if len(v) == 0:
                    continue
                new_relation_item.append([k[0], k[1], max(v)])
            new_relation.append(np.array(new_relation_item))
        return new_relation

    def get_statistics(self):
        _, bg_matrix, pred_matrix = get_VG_statistics(self, must_overlap=True)
        fg_matrix = get_VG_statistics_wo_sample(self)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)
        down_rate, generate_rate = None, None
        if self.cfg.MODEL.STAGE == "stage2" and self.split == 'train':
            down_rate, generate_rate = down_sample_num(pred_matrix, self.freq, self.prob, self.logger, 0.9)
        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'pred_matrix': pred_matrix,
            'data_length': len(self.idx_list),
            'down_rate': down_rate,
            'generate_rate': generate_rate,
            'repeat_dict': self.repeat_dict,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)
        target.add_field("image_index", index)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        return len(self.idx_list)


def filter_dup_relations(input_relation):
    if len(input_relation) == 0:
        return input_relation
    tpt_set = set()
    filtered_relations = []
    for item in input_relation:
        item_tuple = (item[0], item[1], item[2])
        if item_tuple in tpt_set:
            continue
        tpt_set.add(item_tuple)
        filtered_relations.append(item)
    return np.array(filtered_relations)


def get_VG_statistics(train_data, must_overlap=True):
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
    whole_bg, lap_bg, lap_fg = 0, 0, 0
    pred_matrix = np.zeros(num_rel_classes, dtype=np.int64)
    for ex_ind in tqdm(range(len(train_data))):
        if train_data.repeat_dict is not None:
            ex_ind = train_data.idx_list[ex_ind]
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        gt_relations = filter_dup_relations(gt_relations)
        # For the foreground, we'll just look at everything
        o1o2_set = set()
        o1o2 = gt_classes[gt_relations[:, :2]]
        o1o2_ind = gt_relations[:, :2]
        for (o1, o2), gtr, (o1_ind, o2_ind) in zip(o1o2, gt_relations[:, 2], o1o2_ind):
            o1o2_set.add((int(o1_ind), int(o2_ind)))
            fg_matrix[o1, o2, gtr] += 1
            pred_matrix[gtr] += 1
        # For the background, get all of the things that overlap.
        lap_o1o2_ind = np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)
        for (o1_ind, o2_ind) in lap_o1o2_ind:
            if (int(o1_ind), int(o2_ind)) in o1o2_set:
                lap_fg += 1
            else:
                lap_bg += 1
        o1o2_total = gt_classes[lap_o1o2_ind]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1
            pred_matrix[0] += 1
        whole_bg += (len(gt_classes) * (len(gt_classes) - 1) - len(gt_relations))
    fg = np.sum(fg_matrix)
    train_data.logger.info(
        "FG per non-lap BG: {:.5f}, FG per BG: {:.5f}".format(fg / (whole_bg - lap_bg), fg / whole_bg))
    train_data.logger.info("lap FG per FG: {:.5f}, lap FG per lap BG: {:.5f}".format(lap_fg / fg, lap_fg / lap_bg))
    train_data.logger.info("non-lap FG per non-lap BG: {:.5f}".format((fg - lap_fg) / (whole_bg - lap_bg)))
    return fg_matrix, bg_matrix, pred_matrix


def get_VG_statistics_wo_sample(train_data):
    # 这里的fg matrix经过重采样了，做一个没经过重采样的。
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    for ex_ind in tqdm(range(len(train_data.relationships))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_relations = filter_dup_relations(gt_relations)
        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
    return fg_matrix


def get_sample_rate(train_data):
    num_rel_classes = len(train_data.ind_to_predicates)
    src_list = np.zeros(num_rel_classes, dtype=np.float32)
    sampled_list = np.zeros(num_rel_classes, dtype=np.float32)
    result_list = np.zeros(num_rel_classes, dtype=np.float32)
    for ex_ind in tqdm(range(len(train_data))):
        if train_data.repeat_dict is not None:
            ex_ind = train_data.idx_list[ex_ind]
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_relations = filter_dup_relations(gt_relations)
        # For the foreground, we'll just look at everything
        for gtr in zip(gt_relations[:, 2]):
            sampled_list[gtr] += 1
    for ind in tqdm(range(len(train_data.relationships))):
        gt_relations = train_data.relationships[ind].copy()
        gt_relations = filter_dup_relations(gt_relations)
        # For the foreground, we'll just look at everything
        for gtr in zip(gt_relations[:, 2]):
            src_list[gtr] += 1
    # 看采样比例，采样比例增加了就增广。
    median = np.median(sampled_list[1:])
    sampled_num = np.zeros(len(sampled_list))
    for i in range(1, len(sampled_num)):
        if sampled_list[i] > median:
            rate = 1.0 * median / sampled_list[i]
            if rate < 0.01:
                rate = 0.01
            sampled_num[i] = int(sampled_list[i] * rate)
        else:
            sampled_num[i] = sampled_list[i]
    sampled_rate = sampled_num / sum(sampled_num)
    src_rate = src_list / sum(src_list)
    # result_list[1:] = (sampled_rate[1:] - src_rate[1:]) / sampled_rate[1:]
    result_list[1:] = (sampled_num[1:] - src_list[1:]) / sampled_num[1:]
    result_list[result_list < 0] = 0.0
    return result_list


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap,
                zs_file=None, filter_entity_nms=False, nms_thr=0.9):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
        zs_file: for zero-shot
        filter_entity_nms: If training, filter entities that overlap by 0.x IOU.
        nms_thr: thr IOU
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    # split = 'train'
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag
    # split_mask = data_split != -1

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if (num_im > 0) and (split == 'train'):
        image_index = image_index[:-num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]

    if zs_file is not None and split == 'train':
        zs_tmp = torch.load(zs_file).numpy().tolist()
        zs_tpt = {(x[0], x[1], x[2]) for x in zs_tmp}
    else:
        zs_tpt = None

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            # we rearrange the order of the label, from the predicate who owns maximum training samples to the fewer
            new_pre_list = []
            for pre_raw in predicates:
                new_pre_list.append(predicate_new_order[pre_raw])
            predicates = new_pre_list
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        if filter_entity_nms:
            assert split == 'train'
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inter_iou = boxlist_iou(boxes_i_obj, boxes_i_obj)
            inter_cls = gt_classes_i[:, None] == gt_classes_i[None, :]
            inters = (inter_iou > nms_thr) and inter_cls
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
    return split_mask, boxes, gt_classes, relationships
