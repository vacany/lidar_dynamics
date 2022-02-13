import pptk
import numpy as np
import sys
from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
from data.datasets.once.once import Once_dataset

from exps.rw import load_yaml

dataset_choice = int(sys.argv[1])
sequence = int(sys.argv[2])
frame = int(sys.argv[3])
vis_class = sys.argv[4]


# config = load_yaml(config_path)

if dataset_choice == 0:
    config = load_yaml('../semantic-kitti.yaml')
    dataset = SemKittiDataset(prev=config['PEDESTRIAN']['NBR_OF_FRAMES'], sequence=[sequence])
    sequence = f"{sequence:02d}"
    batch = dataset.get_multi_frame(frame)
    file = f"/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/experiments/run_semantic-kitti/data/{sequence:02d}/gen_labels/{frame:06d}.npz"

elif dataset_choice == 1:
    config = load_yaml('../once.yaml')
    sequence = f"{sequence:06d}"
    dataset = Once_dataset(sequences=[sequence])
    batch = dataset.get_sample(frame)
    file = f"/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/experiments/run_once/data/{sequence}/gen_labels/{frame:06d}.npz"

# dataset = SemKittiDataset(prev=config['PEDESTRIAN']["NBR_OF_FRAMES"], sequence=[sequence])
# file = f"/home/patrik/mnt/hdd/iros_2022/run_semantic-kitti/data/{sequence:02d}/gen_labels/{frame:06d}.npz"


pcl = batch['points']
results = np.load(file, allow_pickle=True)

res = results['arr_0'].item()

print(res.keys())

pcl = pcl[pcl[:,4] == frame]
label = res[int(frame)]['seg_label']
# label = res[int(frame)][vis_class.upper()]

if res[int(frame)]['boxes'] is not None:
    ### Check for boxes
    boxes = res[int(frame)]['boxes']
    print(boxes)
    box_ped_mask = (boxes[:,7] == 2) & (boxes[:,3] < 0.8) & (boxes[:,4] < 0.8)
    # boxes = boxes[box_ped_mask]

    from data.point_clouds.box import concatenate_box_pcl
    pcl, label = concatenate_box_pcl(boxes, pcl, label, box_label=100)


#TODO DO THIS ONCE AND CONSISTENTLY!!! - write it down first
# RIGHT WAY TO ASSIGN LABELS!!!
def assign_label_by_ind(seg, pcl_ind, prediction, label_number):
    seg_ = seg.copy()
    seg_[pcl_ind[prediction]] = label_number
    return seg_


v=pptk.viewer(pcl[:,:3], label)
v.set(point_size=0.02)
