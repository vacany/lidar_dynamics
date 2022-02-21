import glob
import os
import json
from exps.utils import timeit
import numpy as np
from scipy.spatial.transform import Rotation

import socket
host = socket.gethostname()

if 'Patrik' == host:
    import pptk
    DATASET_PATH = '/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/once/'

else:
    DATASET_PATH = '/mnt/beegfs/gpu/temporary/vacekpa2/once/'


class Once_dataset():
    def __init__(self, splits=['train'], sequences='all', max_data=10000000, n_sweeps=0):
        self.data_root = DATASET_PATH
        self.splits = splits
        self.sequences = sequences
        self.max_data = max_data
        self.n_sweeps = n_sweeps

        if n_sweeps > 0:
            raise ValueError("MORE SWEEPS ARE NOT IMPLEMENTED")

        self.label_config = {'Car': 1,
                             'Pedestrian': 2,
                             'Cyclist': 3,
                             'Bus': 4,
                             'Truck': 5}

        self.ignore_clz = ['Bus', 'Truck']

        self.gather_sequences()

    def gather_sequences(self):
        self.load_splits()
        self.meta_data = {}
        self.samples = {}
        frame = 0

        for key, sequences in self.sequences_by_split.items():

            for sequence in sequences:

                lidar_paths = glob.glob(f"{self.data_root}/data/{sequence}/lidar_roof/*.bin")
                anno = self.load_sequence(sequence)


                self.meta_data[sequence] = {'lidar_paths': lidar_paths,
                                            'anno': anno,
                                            }

                poses = self.get_seq_poses(sequence)

                self.meta_data[sequence]['poses'] = poses

                for id in range(len(anno['frames'])):
                    self.samples[frame] = {'seq_id' : sequence, 'seq_frame' : id}
                    frame += 1

        self.data = self.samples

    def __len__(self):
        return len(self.samples)

    def get_multi_frame(self, index):
        ''' Main method to give data to the dataloader '''
        data_id = self.samples[index]
        data = self.get_multi_pcl(data_id['seq_id'], data_id['seq_frame'])
        return data

    @timeit
    def get_multi_pcl(self, seq_id, seq_frame):
        anno = self.meta_data[seq_id]['anno']
        # Boxes only for reference - last one
        poses = []
        pcls = []

        for i in range(-self.n_sweeps, 1):
            if seq_frame + i < 0:
                continue
            try:
                pcl = self.get_pcl(seq_id, seq_frame + i, anno=anno)
                pcl = np.insert(pcl, 4, i, axis=1)

                pose = self.meta_data[seq_id]['poses'][seq_frame + i]

                pcls.append(pcl)
                poses.append(pose)

            except:
                pass



        global_pcl = []
        for i in range(len(pcls)):

            cur_points = np.insert(pcls[i][:, :3], 3, 1, axis=1)

            global_points = (cur_points @ poses[i].T)[:, :3]

            global_points = np.concatenate((global_points[:,:3], pcls[i][:,3:]), axis=1)

            global_points[:, 4] = seq_frame + i
            global_pcl.append(global_points)

        global_pcl = np.concatenate(global_pcl)

        local_points = global_pcl.copy()

        local_points[:,:3] -= pose[:3,-1]



        # TODO Still local coords
        boxes = self.get_boxes(seq_id, seq_frame=seq_frame, anno=anno)

        if boxes is not None:
            boxes = np.stack(boxes)

        data = {'points' : global_pcl,
                'local_points' : local_points,
                'boxes' : boxes,
                'poses' : poses,
                'seg_label' : None,
                'instance_label' : None,
                'seq_nbr' : seq_id,
                'frame' : seq_frame
                }

        return data

    def load_splits(self):
        splits = os.listdir(self.data_root + 'ImageSets/')

        self.sequences_by_split = {}
        for split in splits:
            # remove .txt
            split = split[:-4]

            if split in self.splits:

                with open(self.data_root + '/ImageSets/' + split + '.txt', 'r') as f:
                    txt_sequences = f.readlines()

                    txt_sequences = [txt_sequences[i].strip() for i in range(len(txt_sequences))]

                    if self.sequences == 'all':
                        pass
                    else:
                        txt_sequences = [i for i in txt_sequences if i in self.sequences]

                    self.sequences_by_split[split] = txt_sequences


    def load_sequence(self, seq_id):
        anno_file = self.data_root + 'data/' + seq_id + f"/{seq_id}.json"
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        return anno

    def get_boxes(self, seq_id, seq_frame, anno=None):
        if anno is None:
            anno = self.load_sequence(seq_id)

        if 'annos' not in anno['frames'][seq_frame].keys():
            return None

        classes = anno['frames'][seq_frame]['annos']['names']
        anno_boxes = anno['frames'][seq_frame]['annos']['boxes_3d']

        bbox_list = []
        for i in range(len(classes)):

            if classes[i] in self.ignore_clz:
                continue

            box = np.array(anno_boxes[i])
            cls = self.label_config[classes[i]]

            box = np.insert(box, 7, cls, axis=0)

            bbox_list.append(box)

        boxes = np.stack(bbox_list)

        return boxes

    def get_seg_label(self, seq_id, seq_frame):
        return -1

    def get_pose(self, seq_id, seq_frame):
        anno = self.meta_data[seq_id]['anno']

        transform_data = anno['frames'][seq_frame]['pose']
        rotation = Rotation.from_quat(transform_data[:4]).as_matrix()
        translation = np.array(transform_data[4:]).transpose()

        T = np.ones((4, 4))
        T[:3, :3] = rotation
        T[:3, -1] = translation
        Trans_mat = T

        return Trans_mat

    def get_seq_poses(self, seq_id):
        anno = self.load_sequence(seq_id)

        nbr_of_frames = len(anno['frames'])
        all_poses = [self.get_pose(seq_id, seq_frame=i) for i in range(nbr_of_frames)]
        all_poses = np.stack(all_poses)

        return all_poses

    def get_pcl(self, seq_id, seq_frame, anno=None):
        if anno is None:
            anno = self.load_sequence(seq_id)

        frame_name = anno['frames'][seq_frame]['frame_id']

        bin_path = f"{self.data_root}/data/{seq_id}/lidar_roof/{frame_name}.bin"
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    # def get_multi_pcl(self):

#TODO Dont forget, that ONCE has missing annotations, might cause problem

if __name__ == '__main__':
    #TODO Presentation for Patrick Perez - video?
    #TODO

    #TODO boxes to global coordinate system
    data_root = '/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/once/'
    sequence = ['000113', '000027']
    dataset = Once_dataset(splits=['train'], sequences=sequence, n_sweeps=0)
    dataset.gather_sequences()
    batch = dataset.get_sample(20)
    pcl = batch['local_points'][:,:5]

    if host == 'Patrik' and False:
        if batch['boxes'] is not None:
            boxes = batch['boxes']
            from data.point_clouds.box import concatenate_box_pcl, connect_3d_corners

            corners = connect_3d_corners(boxes, add_label=1)
            pcl, label = concatenate_box_pcl(corners, batch['local_points'], label=np.zeros(pcl.shape[0]))

        v=pptk.viewer(pcl[:,:3], label)
        v.set(point_size=0.02)

    else:

        v = pptk.viewer(pcl[:,:3], (pcl[:,3] / 255) > 0.3)
        v.set(point_size=0.02)


