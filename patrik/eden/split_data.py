import os.path

import pptk
import numpy as np
import sys
from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
from exps.rw import load_yaml

frame = 7
sequence = 7
config_path = '/home/patrik/mnt/hdd/iros_2022/run_semantic-kitti/config.yaml'
config = load_yaml(config_path)

dataset = SemKittiDataset(prev=config['PEDESTRIAN']["NBR_OF_FRAMES"], sequence=[sequence])

batch = dataset.get_multi_frame(frame)

# file = f"/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/run{run}/{sequence:02d}/gen_labels/{frame:06d}.npz"
file = f"/home/patrik/mnt/hdd/iros_2022/run_semantic-kitti/data/{sequence:02d}/gen_labels/{frame:06d}.npz"

### Reading file
res = np.load(file, allow_pickle=True)

# def meta_to_seg_label(data_dict):

def assign_label_by_ind(seg, pcl_ind, prediction, label_number):
    seg_ = seg.copy()
    seg_[pcl_ind[prediction]] = label_number
    return seg_

def get_data_dict_from_results(res):

    # functions to work with batch
    times = batch['points'][:, 4]

    time_list = np.unique(times)

    new_data = {}

    for t in time_list:

        cur_time = times == t
        tmp_data = {}
        seg_label =  - np.ones(cur_time.shape)

        cls_list = ['BUILDING', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'TRAFFIC_SIGN']
        for label_nbr, cls in enumerate(cls_list):

            object_ids = res[cls]
            point_cloud = batch['points']
            label = - np.ones(point_cloud.shape[0])

            if cls == 'PEDESTRIAN':
                label[res['ped_valid_indices']] = object_ids
            elif cls == 'TRAFFIC_SIGN':
                label[res['cluster_valid_indices']] = np.array(res[cls], dtype=np.int)
            else:
                label[res['cluster_valid_indices']] = object_ids

            label = label[cur_time]

            seg_label[label >= 0] = label_nbr

            tmp_data[cls] = label
            tmp_data['seg_label'] = seg_label

        new_data[int(t)] = tmp_data


    return new_data

def strip_dict_item(npz_file):
    return npz_file['arr_0'].item()




new_data = get_data_dict_from_results(res)

path_per_frame = os.path.dirname(file) + '/../_per_frame/' + os.path.basename(file)
np.savez_compressed(path_per_frame, new_data)


loaded_data = strip_dict_item(np.load(path_per_frame, allow_pickle=True))


print('done')
