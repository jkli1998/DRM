import json
import logging
import os
import pickle
import random
from collections import defaultdict, OrderedDict, Counter

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.visual_genome import get_VG_statistics, get_sample_rate
from .sample import median_resampling_dict_generation
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.comm import get_rank, synchronize
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, split_boxlist


def load_cate_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    ind_to_predicates_cate = ['__background__'] + info['rel']
    ind_to_entites_cate = ['__background__'] + info['obj']

    # print(len(ind_to_predicates_cate))
    # print(len(ind_to_entites_cate))
    predicate_to_ind = {idx: name for idx, name in enumerate(ind_to_predicates_cate)}
    entites_cate_to_ind = {idx: name for idx, name in enumerate(ind_to_entites_cate)}

    return (ind_to_entites_cate, ind_to_predicates_cate,
            entites_cate_to_ind, predicate_to_ind)


def load_annotations(annotation_file, img_dir, num_img, split,
                     filter_empty_rels, ):
    """

    :param annotation_file:
    :param img_dir:
    :param img_range:
    :param filter_empty_rels:
    :return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    annotations = json.load(open(annotation_file, 'r'))

    if num_img == -1:
        num_img = len(annotations)

    annotations = annotations[: num_img]

    empty_list = set()
    if filter_empty_rels:
        for i, each in enumerate(annotations):
            if len(each['rel']) == 0:
                empty_list.add(i)
            if len(each['bbox']) == 0:
                empty_list.add(i)

    print('empty relationship image num: ', len(empty_list))

    boxes = []
    gt_classes = []
    relationships = []
    img_info = []
    for i, anno in tqdm(enumerate(annotations)):

        if i in empty_list:
            continue

        boxes_i = np.array(anno['bbox'])
        gt_classes_i = np.array(anno['det_labels'], dtype=int)

        rels = np.array(anno['rel'], dtype=int)

        gt_classes_i += 1
        rels[:, -1] += 1

        image_info = {
            'width': anno['img_size'][0],
            'height': anno['img_size'][1],
            'img_fn': os.path.join(img_dir, anno['img_fn'] + '.jpg')
        }

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        img_info.append(image_info)

    return boxes, gt_classes, relationships, img_info


class OIDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split, img_dir, ann_file, cate_info_file, transforms=None,
                 num_im=-1, check_img_file=False, filter_duplicate_rels=True, flip_aug=False):
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
        #
        # num_im = 20000
        # num_val_im = 1000

        assert split in {'train', 'val', 'test'}
        self.cfg = cfg
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.cate_info_file = cate_info_file
        self.annotation_file = ann_file
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = check_img_file
        self.remove_tail_classes = False

        self.ind_to_classes, self.ind_to_predicates, self.classes_to_ind, self.predicates_to_ind = load_cate_info(
            self.cate_info_file)  # contiguous 151, 51 containing __background__

        logger = logging.getLogger("maskrcnn_benchmark.dataset")
        self.logger = logger

        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.gt_boxes, self.gt_classes, self.relationships, self.img_info, = load_annotations(
            self.annotation_file, img_dir, num_im, split=split,
            filter_empty_rels=False if not cfg.MODEL.RELATION_ON and split == "train" else True,
        )

        self.filenames = [img_if['img_fn'] for img_if in self.img_info]
        self.idx_list = list(range(len(self.filenames)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}

        if cfg.MODEL.STAGE == "stage2" and self.split == 'train':
            # creat repeat dict in main process, other process just wait and load
            if get_rank() == 0:
                repeat_dict = median_resampling_dict_generation(self, cfg, self.ind_to_predicates, logger)
                self.repeat_dict = repeat_dict
                with open(os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl"), "wb") as f:
                    pickle.dump(self.repeat_dict, f)

            synchronize()
            self.repeat_dict = median_resampling_dict_generation(self, cfg, self.ind_to_predicates, logger)

            duplicate_idx_list = []
            for idx in range(len(self.filenames)):
                r_c = self.repeat_dict[idx]
                duplicate_idx_list.extend([idx for _ in range(r_c)])
            self.idx_list = duplicate_idx_list

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
            pre_comp_result = torch.load(os.path.join(self.cfg.DATASETS.OIV6_BBOX_DIR, fp_name))
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

    def get_statistics(self):
        fg_matrix, bg_matrix, pred_matrix = get_VG_statistics(self, must_overlap=True)
        over_sampled_rate = get_sample_rate(self)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)
        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'pred_matrix': pred_matrix,
            'data_length': len(self.idx_list),
            'over_sampled': over_sampled_rate,
            'predicate_new_order_count': [],
        }
        return result

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        # if not inner_idx:
        #     # here, if we pass the index after resampeling, we need to map back to the initial index
        #     if self.repeat_dict is not None:
        #         index = self.idx_list[index]
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        box = self.gt_boxes[index]
        box = torch.from_numpy(box)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(np.zeros((len(self.gt_classes[index]), 10))))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)

        relation_map = torch.zeros((num_box, num_box), dtype=torch.long)
        for i in range(relation.shape[0]):
            # Sometimes two objects may have multiple different ground-truth predicates in VisualGenome.
            # In this case, when we construct GT annotations, random selection allows later predicates
            # having the chance to overwrite the precious collided predicate.
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] != 0:
                if random.random() > 0.5:
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
