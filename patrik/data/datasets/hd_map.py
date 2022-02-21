import numpy as np
import matplotlib.pyplot as plt

from data.datasets.semantic_kitti.semantic_kitti import get_ego_bbox
from data.point_clouds.grid import bev


class HD_map():
    '''
    HD map on coordinate level. Used for DA structure and Sidewalk structure. Further for label propagation
    '''
    def __init__(self, cell_size=(0.3,0.3), odo=None):
        if odo is None:
            self.x_range = (-100, 100)
            self.y_range = (-100, 100)
        else:
            x_min = odo[:, 0].min() - 300
            x_max = odo[:, 0].max() + 300
            y_min = odo[:, 1].min() - 300
            y_max = odo[:, 1].max() + 300

            self.x_range = (int(x_min-1), int(x_max+1))
            self.y_range = (int(y_min-1), int(y_max+1))


        self.cell_size = cell_size

        self.hd_map_shape = bev.calculate_shape(self.x_range, self.y_range, cell_size)
        # print(self.hd_map_shape.shape)
        # Types of maps dynamically
        self.hd_map = np.zeros(self.hd_map_shape) - 1
        # self.da_grid = np.zeros(self.hd_map_shape) - 1
        # self.info_grid = np.zeros(self.hd_map_shape) - 1
        # self.height_grid = np.zeros(self.hd_map_shape) - 1

        self.surround_map = self.hd_map.copy()

        self.all_bounding_boxes = []

        self.ego_position = bev.ego_position(x_range=self.x_range, y_range=self.y_range, cell_size=self.cell_size)
        self.ego_bbox = get_ego_bbox()
        self.under_ego_margin = - self.ego_bbox[5] + 0.1

    def calculate_pcl_xy_coordinates(self, pcl, mask=True):
        if mask:
            pcl_mask = bev.mask_out_of_range_coors(pcl, x_range=self.x_range, y_range=self.y_range)
            pcl = pcl[pcl_mask]

        pcl[:,0] -= self.x_range[0]
        pcl[:,1] -= self.y_range[0]

        xy_pcl = bev.calculate_pcl_xy_coordinates(pcl, cell_size=self.cell_size)

        if len(xy_pcl) > 0:
            if xy_pcl.min() < 0:
                import matplotlib.pyplot as plt
                plt.plot(xy_pcl[:, 0], xy_pcl[:, 1], 'b.')
                plt.show()
                raise ValueError('points are in negative coordinates!')

        return xy_pcl

    def DA_by_Ego(self, traj_cls):
        ego_pcl = self.ego_points()
        for odo in traj_cls.all_poses:
            pcl = np.insert(ego_pcl, 3, 1, axis=1)
            pcl = pcl @ odo.T

            self.store_features(pcl, pcl[:,2], grid_name='height_grid')
            self.store_features(pcl, 1, grid_name='da_grid')


    def surrounding_area(self, reference_pose, last_poses):
        ego_pcl = self.ego_points(dl=0, dw=0, dh=0)

        from scipy.spatial.transform import Rotation


        for odo in last_poses:
            pcl = np.insert(ego_pcl, 3, 1, axis=1)
            pcl = pcl @ odo.T

            # TO reference position
            pcl[:,:3] -= reference_pose[:3, -1]
            # Rotate along z-axis
            for yaw in range(0, 180, 3):
                rot_yaw = yaw / 180 * np.pi
                rot_mat = Rotation.from_rotvec(np.array((0,0,rot_yaw)))
                rotated_pcl = pcl[:,:3] @ rot_mat.as_matrix()
                # Back to original position
                rotated_pcl += reference_pose[:3, -1]

                self.store_features(rotated_pcl, rotated_pcl[:,2], grid_name='surround_grid', method='max')


    def ego_points(self, dl=0, dw=0, dh=0):
        #TODO Refactor somewhere else
        '''

        :return: point cloud of ego
        '''
        x, y, z, l, w, h, yaw = self.ego_bbox

        l += dl
        w += dw
        h += dh
        safe_margin = 0.5

        pcl_list = []
        for i in range(int(l / self.cell_size[0] / safe_margin)):
            for j in range(int(w / self.cell_size[1] / safe_margin)):
                pcl_list.append(np.array((i * self.cell_size[0] * safe_margin, j * self.cell_size[1] * safe_margin, 0), dtype=np.float))

        pcl = np.stack(pcl_list)

        xy_shift = np.array((l / 2 , w / 2, 0), dtype=np.float)

        pcl -= xy_shift

        return pcl

    def store_hd_map(self, path):
        config = {'cell_size' : self.cell_size, 'x_range' : self.x_range, 'y_range' : self.y_range}

        for attr in dir(self):
            if 'grid' in attr:
                hd_mapa = getattr(self, attr)
                config[attr] = hd_mapa

        np.savez(path, **config)

    def load_hd_map(self, path):
        hd = np.load(path, allow_pickle=True)

        self.cell_size = hd['cell_size']
        self.x_range = hd['x_range']
        self.y_range = hd['y_range']

        for attr in hd.files:
            if 'grid' in attr:
                setattr(self, attr, hd[attr])

    def get_ego_surrounding(self, pcl):
        '''
        :param surround_map: Extended height map from the ego-motion
        :return:
        '''
        xy_pcl = self.calculate_pcl_xy_coordinates(pcl)
        sa_mask = self.surround_map[xy_pcl[:, 0], xy_pcl[:, 1]] != -1
        ground_mask = pcl[:, 2] > self.surround_map[xy_pcl[:, 0], xy_pcl[:, 1]] + self.under_ego_margin

        return ground_mask, sa_mask

    def DA_containing_points(self, pcl):
        xy = self.calculate_pcl_xy_coordinates(pcl)

        on_DA = self.da_grid[xy[:,0], xy[:,1]] == 1

        return on_DA

    def points_above_DA(self, pcl):
        xy = self.calculate_pcl_xy_coordinates(pcl)
        on_DA = self.DA_containing_points(pcl)

        above_DA = self.height_grid[xy[:,0], xy[:,1]] + self.under_ego_margin < pcl[:,2]

        points_moving_on_DA = on_DA * above_DA

        return points_moving_on_DA

    def __valid_name(self, grid_name):
        if 'grid' not in grid_name:
            raise ValueError("All maps should have 'grid' in variable name")

    def __return_map_names(self):
        return [grid_name for grid_name in dir(self) if 'grid' in grid_name]


    def store_features(self, pcl, features, grid_name, method='overwrite'):
        '''
        :param grid_name: Assign values to the specific grid variable
        :param method: how it should be assigned (overwrite, add ...)
        :return: Store values inside the map, does not return
        '''
        self.__valid_name(grid_name)

        if not hasattr(self, grid_name):
            setattr(self, grid_name, np.zeros(self.hd_map.shape))

        xy = self.calculate_pcl_xy_coordinates(pcl)


        if method == 'overwrite':
            getattr(self, grid_name)[xy[:,0], xy[:,1]] = features

        elif method == 'add':
            getattr(self, grid_name)[xy[:, 0], xy[:, 1]] += features

        elif method == 'max':
            sort_mask = features.argsort()
            xy = xy[sort_mask]
            features = features[sort_mask]

            # keep previous maximum
            tmp_features = getattr(self, grid_name)[xy[:, 0], xy[:, 1]]
            features[features < tmp_features] = tmp_features[features < tmp_features]

            getattr(self, grid_name)[xy[:, 0], xy[:, 1]] = features

        elif method == 'min':
            sort_mask = features.argsort()
            xy = xy[sort_mask]
            features = features[sort_mask]

            # keep previous maximum
            tmp_features = getattr(self, grid_name)[xy[:, 0], xy[:, 1]]
            features[features < tmp_features] = tmp_features[features < tmp_features]

            getattr(self, grid_name)[xy[:, 0], xy[:, 1]] = features

        elif method == 'new':
            setattr(self, grid_name, np.zeros(self.hd_map.shape))
            getattr(self, grid_name)[xy[:, 0], xy[:, 1]] = features

    def transfer_features(self, pcl, grid_name):
        '''
        :return: Values from the map on the point cloud coordinates
        '''
        self.__valid_name(grid_name)

        xy = self.calculate_pcl_xy_coordinates(pcl)
        values = getattr(self, grid_name)[xy[:, 0], xy[:, 1]]

        return values

    def show_feature_maps(self):
        grid_names = self.__return_map_names()
        nbr_of_maps = len(grid_names)

        fig, axes = plt.subplots(nrows=1, ncols=nbr_of_maps)
        for idx, grid in enumerate(grid_names):
            hd_map = getattr(self, grid)

            if nbr_of_maps == 1:
                axes.imshow(hd_map)
                axes.set_title(grid)

            else:
                axes[idx].imshow(hd_map)
                axes[idx].set_title(grid)

        plt.show()






