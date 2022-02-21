import numpy as np
import os

from data.point_clouds.clustering import structure
from data.datasets.hd_map import HD_map
from exps.rw import load_yaml





class Annotator():
    def __init__(self, config=None, hd_map=HD_map()):
        self.config = config
        self.hd_map = hd_map

    def update(self, batch):
        ''' make smoother? just visual problem, works anyway ... '''
        trans = batch['all_poses'][batch['frame']]

        local_pcl = batch['points'].copy()
        inten = local_pcl[:,3].copy()

        local_pcl[:,3] = 1

        local_pcl[:,:3] = (np.linalg.inv(trans) @ local_pcl[:,:4].T)[:3, :].T
        # import pptk
        # pptk.viewer(local_pcl[:,:3])
        local_pcl[:,3] = inten
        self.local_pcl = local_pcl


        self.global_pcl = batch['points']

        # indices = np.argsort(self.local_pcl[:,2])
        # self.local_pcl

        self.instance_seg = np.zeros(self.local_pcl[:,0].shape, dtype=np.int) - 1
        self.seg = np.zeros(self.local_pcl[:,0].shape, dtype=np.int) - 1 # + self.config['clz_names'][6] #outlier

    def __above_ground(self):
        mask = self.local_pcl[:, 2] > self.hd_map.under_ego_margin + self.config['add_ground_height']
        return mask

    def frame_clustering(self):
        ''' Get intensity mask of vehicles '''
        self.inten_mask = structure.get_licence_plates(self.local_pcl,
                                                       inten_threshold=self.config['inten_threshold'],
                                                       margin=self.hd_map.under_ego_margin / 2)

        ''' Get clusters with those peak values of intensity '''
        self.cluster_mask, self.all_clusters = structure.get_unique_clusters(self.local_pcl,
                                                          self.local_pcl[:, 2] > self.hd_map.under_ego_margin +
                                                          self.config['add_ground_height'],
                                                          true_mask=self.inten_mask,
                                                          min_dist=self.config['cluster_distance'],
                                                          min_samples=self.config['min_samples_inten'])

        ''' Sanity check and additional heuristics '''
        self.cluster_mask = structure.filter_building(self.local_pcl, self.cluster_mask, max_size=self.config['building_min_dist'])

        # self.cluster_mask[instance_seg == -1] = self.config['clz_names']['Background']
        self.cluster_mask[self.cluster_mask > 0] = self.config['clz_names']['Inten_Cluster']

        # Assign instances
        # self.instance_seg = self.cluster_mask

        self.seg[self.cluster_mask > 0] = self.cluster_mask[self.cluster_mask > 0]

    def store_moving_cls(self):
        ''' Store moving objects to hd map '''
        moving_mask = self.seg == self.config['clz_names']['Vehicle']

        above_ground = self.__above_ground()

        moving_points = self.global_pcl[moving_mask * above_ground]

        self.hd_map.store_features(moving_points, features=1, grid_name='moving_objects_grid')

    def transfer_moving_cls(self):
        above_ground = self.__above_ground()

        old_objects = self.hd_map.transfer_features(self.global_pcl, grid_name='moving_objects_grid')

        previous_objects = old_objects * above_ground

        self.seg[previous_objects==1] = self.config['clz_names']['Vehicle']

    def DA_from_moving_objects(self):
        above_ground = self.__above_ground()

        self.hd_map.store_features(self.global_pcl, features=above_ground, grid_name='tmp_high_occ_grid', method='new')

        self.hd_map.trace_grid = self.hd_map.moving_objects_grid * self.hd_map.tmp_high_occ_grid

        potential_DA = self.hd_map.transfer_features(self.global_pcl, grid_name='trace_grid')

        self.hd_map.store_features(self.global_pcl, features=potential_DA, grid_name='da_from_objects_grid')

        self.seg[potential_DA==1] = self.config['clz_names']['Drivable Area']

    def annotate_by_DA(self):
        # ''' Apply Ego Trajectory '''

        ''' Get masks '''
        moving_on_DA = self.hd_map.points_above_DA(self.global_pcl)
        DA = self.hd_map.DA_containing_points(self.global_pcl)
        ''' Store features'''
        # self.hd_map.store_features(self.global_pcl[DA], features=1, grid_name='ego_grid')
        self.hd_map.store_features(self.global_pcl[moving_on_DA], features=1, grid_name='moving_on_DA_grid')
        ''' Update labels '''
        self.seg[DA] = self.config['clz_names']['Drivable Area']
        self.seg[moving_on_DA] = self.config['clz_names']['Moving_on_DA']

        ''' Store labels of objects on ego traj'''
        self.hd_map.store_features(self.global_pcl[moving_on_DA], features=self.seg[moving_on_DA], grid_name='detection_grid')


    def show_labelling(self):
        import pptk
        # pcl = self.pcl[self.pcl[:,2] > 3]
        # v = pptk.viewer(pcl[:, :3], pcl[:,2] > 0)
        v = pptk.viewer(self.global_pcl[:, :3], self.cluster_mask)
        v.set(point_size=0.02)
        v2 = pptk.viewer(self.global_pcl[:, :3], self.all_clusters)
        v2.set(point_size=0.02)



if __name__ == '__main__':
    from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
    from data.datasets.hd_map import HD_map
    from data.trajectory.trajectory import Trajectory
    dataset = SemKittiDataset()

    batch = dataset.get_multi_frame(1100)

    ego = Trajectory.from_poses(batch['all_poses'])
    hd_map = HD_map(cell_size=(0.3,0.3), odo=ego.odometry)
    hd_map.DA_by_Ego(ego)


    config = load_yaml(os.path.dirname(os.path.abspath(__file__).split('.')[0]) + '/config_values.yaml')

    annotator = Annotator(config=config, hd_map=hd_map)
    annotator.update(batch)

    annotator.frame_clustering()
    annotator.annotate_by_DA()

    annotator.show_labelling()

    annotator.hd_map.show_feature_maps()
