import os
import sys
import h5py
import json
import numpy as np
from collections import defaultdict
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.data.datasets.visual_genome import load_graphs, load_image_filenames
from SHA_GCL_extra.group_chosen_function import predicate_new_order, predicate_new_order_name
BOX_SCALE = 1024  # Scale at which we have the boxes


class SemiDataSplit(object):
    def __init__(self, data_path):
        roidb_file = os.path.join(data_path, "vg/VG-SGG-with-attri.h5")
        img_dir = os.path.join(data_path, "vg/VG_100K")
        image_file = os.path.join(data_path, "vg/image_data.json")
        split_mask, _, self.gt_classes, relationships = load_graphs(
            roidb_file, "train", 0, 5000, filter_empty_rels=True, filter_non_overlap=False,
        )
        self.relationships = self.filter_relations(relationships)
        filenames, img_info = load_image_filenames(img_dir, image_file)
        self.filenames = [filenames[i] for i in np.where(split_mask)[0]]
        self.img_info = [img_info[i] for i in np.where(split_mask)[0]]

        test_split_mask, _, self.test_gt_classes, self.test_relationships = load_graphs(
            roidb_file, "test", 0, 5000, filter_empty_rels=True, filter_non_overlap=False,
        )

        self.test_tpt_set = set()
        for idx in range(len(self.test_relationships)):
            test_cls = self.test_gt_classes[idx].copy()
            test_rel = self.test_relationships[idx].copy()
            o1o2 = test_cls[test_rel[:, :2]]
            for (o1, o2), gtr in zip(o1o2, test_rel[:, 2]):
                self.test_tpt_set.add((int(o1), int(o2), int(gtr)))

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

    def condition_check_all(self):
        # contain all labels.
        name = os.path.join("SEMI_utils", "tmp_rand.npy")
        s_path = os.path.join("SEMI_utils", "split.npy")
        tmp_data = np.load(name, allow_pickle=True).item()
        save_data = defaultdict(dict)
        percents = [1, 2, 5, 10]
        for p in percents:
            cnt = 0
            if p == 1:
                begin, end = 1, 10000
            else:
                begin, end = 1, 1000
            for seed in range(begin, end):
                tmp_data_ind = tmp_data[p][seed]
                if self.condition_check_one(tmp_data_ind):
                    cnt += 1
                    f_name, i_info = [], []
                    for idx in tmp_data_ind:
                        f_name.append(self.filenames[idx])
                        i_info.append(self.img_info[idx])
                    zs_list = self.generate_zero_shot(tmp_data_ind)
                    save_data[p][cnt] = {
                        "seed": seed,
                        "idx": tmp_data_ind,
                        "filenames": f_name,
                        "image_info": i_info,
                        "zs_list": zs_list,
                    }
                    print("{}\t{}".format(p, seed))
                if cnt == 5:
                    break
        np.save(s_path, save_data)

    def condition_check_one(self, data_ind):
        indicator_rel = np.zeros(51, dtype=np.bool_)
        indicator_obj = np.zeros(151, dtype=np.bool_)
        for i, x in enumerate(data_ind):
            gt_relations = self.relationships[x].copy()
            gt_classes = self.gt_classes[x].copy()
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
                indicator_rel[gtr] = True
                indicator_obj[o1] = True
                indicator_obj[o2] = True
        return indicator_rel[1:].all() and indicator_obj[1:].all()

    def generate_zero_shot(self, train_data_ind):
        train_tpt_set = set()
        for idx in train_data_ind:
            train_cls = self.gt_classes[idx].copy()
            train_rel = self.relationships[idx].copy()
            o1o2 = train_cls[train_rel[:, :2]]
            for (o1, o2), gtr in zip(o1o2, train_rel[:, 2]):
                train_tpt_set.add((int(o1), int(o2), int(gtr)))

        zs_set = self.test_tpt_set.difference(train_tpt_set)
        return np.array(list(zs_set))

    def dump_length(self):
        path = os.path.join("SEMI_utils", "relation_length.log")
        with open(path, 'w') as fp:
            fp.write(str(len(self.relationships)))


def main():
    path = sys.argv[1]
    mode = sys.argv[2]
    if mode == 'length':
        SemiDataSplit(path).dump_length()
    else:
        SemiDataSplit(path).condition_check_all()


if __name__ == "__main__":
    main()

