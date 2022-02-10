import pptk
import numpy as np
import sys
from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset

run = int(sys.argv[1])
sequence = int(sys.argv[2])
frame = int(sys.argv[3])

dataset = SemKittiDataset(next=run, chosen_seqs=[sequence])

batch = dataset.get_multi_frame(frame)
file = f"/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/run{run}/{sequence:02d}/gen_labels/{frame:06d}.npz"

# pcl_file = f"/home/patrik/mnt/rci/mnt/data/vras/data/semantic-kitti/dataset/sequences/{sequence:02d}/velodyne/{frame:06d}.bin"
# pcl = np.fromfile(pcl_file, dtype=np.float32)
# pcl = pcl.reshape((-1, 4))

pcl = batch['points']
results = np.load(file, allow_pickle=True)

# print(results.files)

label = results['seg_label']
label = pcl[:,3]
boxes = results['boxes']

from data.point_clouds.box import connect_3d_corners, concatenate_box_pcl
corners = connect_3d_corners(boxes, add_label=7)

pcl, label = concatenate_box_pcl(corners, pcl, label)


# print(pcl.shape, label.shape)
v=pptk.viewer(pcl[:,:3], label)
v.set(point_size=0.02)
