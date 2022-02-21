import numpy as np


from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset

import matplotlib.pyplot as plt
plt.rcParams['image.cmap']='jet'



from exps.utils import timeit

class Visual_BEV():
    def __init__(self, dataset, function, config, offset=0):

        self.dataset = dataset
        self.function = function
        self.offset = offset

        self.original_config = config.copy()
        self.config = config

        self.offset = offset
        self.total = len(dataset)

        # value init
        self.keys = list(self.config.keys())
        self.cur_key = 0
        self.cur_value = self.config[self.keys[self.cur_key]]

        # colorbar
        self.colorbar = False

        self.reset()
        self.update()

    def reset(self):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_title('Use arrow to navigate')

        self.fig.canvas.mpl_connect('key_press_event', self.click)
        plt.show()

    def click(self, event):

        if event.key == 'right':
            self.offset = (self.offset + 1)
        elif event.key == 'left':
            self.offset = (self.offset - 1)
        elif event.key == 'up':
            self.offset = (self.offset + 50)
        elif event.key == 'down':
            self.offset = (self.offset - 50)


        self.update()


    def update(self):
        data = self.dataset.get_multi_frame(self.offset)

        self.config[self.keys[self.cur_key]] = self.cur_value

        bev = self.function(data, self.config)

        self.ax.clear()

        # title
        title = f"{self.keys[self.cur_key]} \t ---> {self.cur_value:.2f} \t Scan {self.offset} out of {self.total}"
        self.ax.set_title(title)

        # print(f"\r \n{self.keys[self.cur_key]}\n \t {self.cur_value:.2f}", end='')


        # plot
        bev[0,0] = 2
        self.im = self.ax.imshow(bev)
        # self.ax.set_clim = (0,6)


        self.im.set_data(bev)
        # self.ax.set_ylabel(f'{self.ind} /  {self.slices}')
        self.im.axes.figure.canvas.draw()

        if not self.colorbar:
            plt.colorbar(self.im, extend='both')

            self.colorbar = True



# config = {'nan' : 0}
from data.datasets.hd_map import HD_map
from data.trajectory.trajectory import Trajectory

dataset = SemKittiDataset(chosen_seqs=[7])
batch = dataset.get_multi_frame(0)
Ego_Traj = Trajectory.from_poses(batch['all_poses'])
odo = Ego_Traj.odometry

hd_map = HD_map(cell_size=(0.2,0.2), odo=odo)
hd_map.store_features(np.zeros((10,3)), features=np.zeros(10), grid_name='bev_grid', method='max')


from data.point_clouds.clustering import driving_scene
from exps.rw import load_yaml
config = load_yaml('../clustering/config_values.yaml')

def clustering_annotator(data, config):

    annotator = driving_scene.Annotator(config=config)
    annotator.update(data)
    annotator.frame_clustering()

    mask = annotator.seg == 1
    inten_mask = mask * (data['points'][:,3] > 0.98)
    rest_points = np.zeros(data['points'][:,3].shape) - 1


    hd_map.store_features(data['points'][:, :3], features=rest_points, grid_name='bev_grid', method='new')

    hd_map.store_features(data['points'][:, :3][mask], features=annotator.seg[mask], grid_name='bev_grid', method='overwrite')
    hd_map.store_features(data['points'][:, :3][inten_mask], features=2 * inten_mask[inten_mask==1], grid_name='bev_grid', method='max')

    bev = hd_map.bev_grid

    bev = bev[1000:1250, 1000:1250]
    # bev = bev[700:900, 700:900]
    return bev


def bev_inten(data, config):
    mask = (data['points'][:,3] > 0.98) * (data['seg_label'] != 6) # outlier
    features = data['seg_label'][mask]

    hd_map.store_features(data['points'][:,:3][mask], features=features, grid_name='bev_grid', method='overwrite')
    print(np.unique(features))

    bev = hd_map.bev_grid


    return bev

def labels(data, config):

    features = np.array(data['seg_label'] == 2, dtype=np.float)
    hd_map.store_features(data['points'][:,:3], features=features, grid_name='bev_grid', method='new')
    print(np.unique(features))

    bev = hd_map.bev_grid
    print(np.argwhere(bev==1))

    bev = bev[2500:2550, 1800:1850]
    return bev

offset=441
batch = dataset.get_multi_frame(offset)

# import pptk
# pptk.viewer(batch['points'][:,:3])

Visual_BEV(dataset=dataset, function=clustering_annotator, config=config, offset=offset)

