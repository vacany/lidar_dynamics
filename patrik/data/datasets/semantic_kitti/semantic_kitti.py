import os.path

import numpy as np
import glob
import socket
from .semantic_lasers import SemLaserScan
from exps.utils import timeit
# from data.trajectory.trajectory import Trajectory

GOEDEL_PATH = '/datagrid/public_datasets/Semantic_Kitti/dataset/sequences/'
PATRIK_PATH = '/home/patrik/mnt/hdd/semantic-kitti/dataset/sequences/'
RCI_PATH = '/mnt/data/vras/data/semantic-kitti/dataset/sequences/'

server = socket.gethostname()

if server == 'goedel':
  DATASET_PATH = GOEDEL_PATH
elif 'rci' in server or len(server) == 3:
  DATASET_PATH = RCI_PATH
elif server == 'Patrik':
  DATASET_PATH = PATRIK_PATH


def get_ego_bbox():
  ### KITTI EGO Parameters
  l = 3.5
  w = 1.8
  h = 1.73
  x, y, z = 0, 0, -h / 2
  angle = 0
  EGO_BBOX = np.array((x, y, z, l, w, h, angle))

  return EGO_BBOX

def parse_calibration(filename):
  """ read calibration file with given filename
      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename
      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = np.linalg.inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses

def load_yaml(path):
  import yaml
  with open(path, 'r') as f:
    return yaml.load(f, Loader=yaml.Loader)

class SemKittiDataset():

  def __init__(self,
               max_samples=10000,
               prev=0,

               next=0,
               every_th=1,
               sync='global',
               sequence=[7],
               infered_labels=None,

               ):

    self.cfg = load_yaml(os.path.dirname(os.path.abspath(__file__)) + '/../config/semantic_kitti.yaml')
    self.dir = DATASET_PATH

    self.infered_labels = infered_labels

    self.max_samples = max_samples
    self.next = next
    self.prev = prev
    self.every_th = every_th
    self.sync = sync


    self.chosen_seqs = sequence

    self.max_label = np.max(list(self.cfg['learning_map'].values()))

    self.laser = SemLaserScan(10, self.cfg['color_map'])

    self.path_points = '/velodyne/'
    self.path_labels = '/labels/'

    for seq in self.chosen_seqs:
      if seq in range(0, 9):
        self.split = 'train'
      elif seq in range(9,11):
        self.split = 'val'
      else:
        self.split = 'test'

    self.gather_sequences()
    self.gather_frames()



  def __len__(self):
    return len(self.data)

  def gather_sequences(self):
    ''' Preprocess all available and right sequences '''
    # Dictionary of helpful variables
    self.meta_data = {}
    # get all sequences
    self.seqs = [seq for seq in sorted(glob.glob(self.dir + '*')) if int(os.path.basename(seq)) in self.chosen_seqs]

    for seq in self.seqs:
      # number
      nbr_of_seq = int(os.path.basename(seq))
      poses = parse_poses(seq + '/poses.txt', parse_calibration(seq + '/calib.txt'))

      self.meta_data[nbr_of_seq] = {'nbr_of_seq' : nbr_of_seq,
                                    'path_seq' : seq,
                                    'poses' : poses,
                                    'lidar_paths' : sorted(glob.glob(seq + self.path_points + '*.bin')),
                                    'label_paths' : sorted(glob.glob(seq + self.path_labels + '*.label'))}



  def gather_frames(self):

    self.data = {}

    ind = 0
    for keys, seq in self.meta_data.items():
        for sample in range(len(seq['lidar_paths'])):

          self.data[ind] = {'lidar_path' : seq['lidar_paths'][sample],
                            'seg_label_path' : seq['label_paths'][sample] if len(seq['label_paths']) > 0 else None,
                            'pose' : seq['poses'][sample],
                            'seq_nbr' : seq['nbr_of_seq'],
                            'seq_path' : seq['path_seq'], # sequence
                            'frame' : ind,
                            'seq_frame' : sample,
                            'bbox' : None,
                            }

          ind += 1

          if ind == self.max_samples:
            return 0



  def reannotate_by_config(self, label):
    learning_map = self.cfg['learning_map']
    for k, v in learning_map.items():
      label[label == k] = v

    return label

  def _get_frame_list(self, ind):
    '''

    :param ind:
    :return: Frames already inside the sequence
    '''
    reference = self.data[ind]['seq_frame']
    indices = range(reference - self.prev, reference + self.next + 1, self.every_th)
    return np.array(indices, dtype=np.int)

  def _gather_frame_global_data(self, ind):
    seq_frames = self._get_frame_list(ind)
    load_pcls = []
    load_poses = []
    load_seg = []
    load_instance = []
    seg_label = None
    instance_label = None

    for ind in seq_frames:

      try:
        lidar_path = os.path.dirname(self.data[ind]['lidar_path']) + f'/{ind:06d}.bin'
        self.laser.open_scan(lidar_path)


        pcl = np.concatenate((self.laser.points, self.laser.remissions[:, None], np.full_like(self.laser.remissions[:,None], ind)), axis=1)

        if self.split != 'test':
          label_path = os.path.dirname(self.data[ind]['seg_label_path']) + f'/{ind:06d}.label'
          self.laser.open_label(label_path)
          instance_label = self.laser.inst_label
          label = self.reannotate_by_config(self.laser.sem_label)
          load_seg.append(label)
          load_instance.append(instance_label)


        load_poses.append(self.data[ind]['pose'])
        load_pcls.append(pcl)


        self.laser.reset()

      except OSError:
        pass

    points = np.concatenate(load_pcls)
    poses = np.stack(load_poses)

    if self.split != 'test':
      seg_label = np.concatenate(load_seg)
      instance_label = np.concatenate(load_instance)


    data = {'points' : points,
            'poses' : poses,
            'seg_label' : seg_label if self.split != 'test' else None,
            'instance_label' : instance_label if self.split != 'test' else None,
            'frame' : ind,
            'seq_frames' : seq_frames}

    return data

  @timeit
  def get_multi_frame(self, frame):

    frame_data = self._gather_frame_global_data(frame)

    # Get multi_cloud
    frame_data = self._get_global_multi_lidar(frame_data)

    all_poses = self.meta_data[self.data[frame]['seq_nbr']]['poses']



    data = {'points' : frame_data['points'],
            'local_points' : frame_data['local_points'],
            'poses' : frame_data['poses'],
            'frame' : frame,

            'seg_label' : frame_data['seg_label'] if self.split != 'test' else None,
            'instance_label' : frame_data['instance_label'] if self.split != 'test' else None,

            'seq_path' : self.data[frame]['seq_path'],
            'seq_nbr' : self.data[frame]['seq_nbr']
            }


    return data

  def _get_global_multi_lidar(self, frame_data):

    points = frame_data['points']

    seq_nbr = self.data[frame_data['frame']]['seq_nbr']
    poses = self.meta_data[seq_nbr]['poses']


    time_frames = frame_data['seq_frames']


    for t in time_frames:
      cur_points = points[points[:,4] == t].copy()

      cur_points = np.insert(cur_points[:, :3], 3, 1, axis=1)

      points[:, :3][points[:,4] == t] = (cur_points @ poses[t].T)[:, :3]


    frame_data['points'] = points

    local_points = frame_data['points'].copy()

    # Sync
    local_points[:, 3] = 1
    reference_trans = poses[frame_data['seq_frames'][-1]]

    local_points[:, :3] = (np.linalg.inv(reference_trans) @ local_points[:, :4].T)[:3, :].T
    local_points[:, 3:] = points[:, 3:].copy()

    frame_data['local_points'] = local_points

    return frame_data

  def strip_frame(self, name):
    frame_str = name.split('/')[-1].split('.')[0]
    return frame_str
