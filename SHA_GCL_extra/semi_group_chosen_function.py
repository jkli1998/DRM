# we rearrange the VG dataset, sort the relation classes in descending order (the original order is based on relation class names)
predicate_new_order = [0, 10, 42, 43, 34, 28, 17, 19, 7, 29, 33, 18, 35, 32, 27, 50, 22, 44, 45, 25, 2, 9, 5, 15, 26, 23, 37, 48, 41, 6, 4, 1, 38, 21, 46, 30, 36, 47, 14, 49, 11, 16, 39, 13, 31, 40, 20, 24, 3, 12, 8]
predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712, 5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352, 663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270, 234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
predicate_new_order_name = ['__background__', 'on', 'has', 'wearing', 'of', 'in', 'near', 'behind', 'with', 'holding', 'above', 'sitting on', 'wears', 'under', 'riding', 'in front of', 'standing on', 'at', 'carrying', 'attached to', 'walking on', 'over', 'for', 'looking at', 'watching', 'hanging from', 'laying on', 'eating', 'and', 'belonging to', 'parked on', 'using', 'covering', 'between', 'along', 'covered in', 'part of', 'lying on', 'on back of', 'to', 'walking in', 'mounted on', 'across', 'against', 'from', 'growing on', 'painted on', 'playing', 'made of', 'says', 'flying in']
import numpy as np

"""
def group_function(predicate_cnt):
    idx = np.argsort(-np.array(predicate_cnt))
    sorted_cnt = np.array(predicate_cnt)[idx]
    incremental_stage_list = []
    tmp_list = []
    p_curr = sorted_cnt[1]
    for i in range(1, len(sorted_cnt)):
        if p_curr > 5 * sorted_cnt[i]:
            p_curr = sorted_cnt[i]
            incremental_stage_list.append(tmp_list)
            tmp_list = []
        tmp_list.append(idx[i])
    predicate_stage_count = [len(x) for x in incremental_stage_list]
    return predicate_stage_count, incremental_stage_list
"""

def group_function(predicate_cnt):
    print("cnt: ", predicate_cnt)

    idx = np.argsort(-np.array(predicate_cnt))

    print("idx: ", idx)

    sorted_cnt = np.array(predicate_cnt)[idx]

    print("sort_cnt: {}".format(len(sorted_cnt)), sorted_cnt)

    incremental_stage_list = []
    tmp_list = []
    p_curr = sorted_cnt[1]
    for i in range(1, len(sorted_cnt)):
        if p_curr > 5 * sorted_cnt[i]:
            p_curr = sorted_cnt[i]
            incremental_stage_list.append(tmp_list)
            tmp_list = []
        tmp_list.append(idx[i])
    incremental_stage_list.append(tmp_list)
    predicate_stage_count = [len(x) for x in incremental_stage_list]
    return predicate_stage_count, incremental_stage_list


def predicate_stage_transfer(incremental_stage_list):
    """transfer a un-sorted stage list to a sorted list (through a mapper list)"""
    # 首先我们需要将输入训练的label转化为对应transfer后的（label换就相当于dists换了），然后需要将最后输出的dists转化回来
    mapper_sort = [0 for _ in range(51)]
    mapper_back = [0 for _ in range(51)]
    new_incremental_stage_list = []
    cnt = 1
    for stage in incremental_stage_list:
        new_stage = []
        for p_idx in stage:
            new_stage.append(cnt)
            mapper_sort[p_idx] = cnt
            mapper_back[cnt] = p_idx
            cnt += 1
        new_incremental_stage_list.append(new_stage)
    return mapper_sort, mapper_back, new_incremental_stage_list


def get_group_splits(Dataset_name, split_name, predicate_cnt, inter_baseline=False):
    assert Dataset_name in ['VG', 'GQA_200']
    incremental_stage_list = None
    predicate_stage_count = None
    if Dataset_name == 'VG':
        if not inter_baseline:
            assert split_name == 'divide4' # [4,4,9,19,12]
            incremental_stage_list = [[1, 2, 3, 4],
                                    [5, 6, 7, 8, 9, 10],
                                    [11, 12, 13, 14, 15, 16, 17, 18, 19],
                                    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                                    [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [4, 6, 9, 19, 12]
            predicate_cnt = predicate_new_order_count
        else:
            predicate_stage_count, incremental_stage_list = group_function(predicate_cnt)
            split_choice = {
                3: 'divide3',
                4: 'divide4',
                5: 'divide5',
            }
            split_name = split_choice[len(predicate_stage_count)]
        print(predicate_stage_count, incremental_stage_list)
        print(predicate_cnt)
        assert sum(predicate_stage_count) == 50

    elif Dataset_name == 'GQA_200':
        assert split_name in ['divide3', 'divide4', 'divide5', 'average']
        if split_name == 'divide3':  # []
            incremental_stage_list = [[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                      [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [4, 4, 11, 16, 31, 34]
        elif split_name == 'divide4':  # [4,4,9,19,12]
            incremental_stage_list = [[1, 2, 3, 4, 5],
                                      [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                      [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [5, 10, 20, 65]
        elif split_name == 'divide5':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7],
                                      [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                                      [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                                      [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [7, 14, 28, 51]
        elif split_name == 'average':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                                      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                                      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [20, 20, 20, 20, 20]
        else:
            exit('wrong mode in group split!')
        assert sum(predicate_stage_count) == 100

    else:
        exit('wrong mode in group split!')
    mapper_sort, mapper_back, incremental_stage_list = predicate_stage_transfer(incremental_stage_list)
    return incremental_stage_list, predicate_stage_count, split_name, mapper_sort, mapper_back, predicate_cnt