import numpy as np
import matplotlib.pyplot as plt

def ego_position(x_range=(0,70), y_range=(-40, 40), cell_size=(0.25, 0.25)):
    ego = (-x_range[0], -y_range[0])
    xy_ego = (ego[0] / cell_size[0], ego[1] / cell_size[1])
    xy_ego = np.array(xy_ego, dtype=np.int)
    return xy_ego

def mask_out_of_range_coors(pcl, x_range=(0, 70), y_range=(-40,40), z_range=(-np.inf, np.inf)):
    '''

    :param pcl: point cloud xyz...
    :param x_range:
    :param y_range:
    :param z_range:
    :return: Mask for each point, if it fits into the range
    '''
    mask = (pcl[:, 0] > x_range[0]) & (pcl[:, 0] < x_range[1]) & \
           (pcl[:, 1] > y_range[0]) & (pcl[:, 1] < y_range[1]) & \
           (pcl[:, 2] > z_range[0]) & (pcl[:, 2] < z_range[1])

    return mask


def calculate_pcl_xy_coordinates(pcl, cell_size=(0.1,0.1)):
    '''

    :param pcl: point cloud xy...
    :param cell_size: size of the bin for point discretization
    :param ego: xy-position of ego in meters
    :return: coordinates for each point in bird eye view
    '''
    xy = np.floor(pcl[:, :2] / cell_size).astype('i4')

    return xy

def calculate_shape(x_range=(0,70), y_range=(-40, 40), cell_size=(0.25, 0.25)):
    '''
    :return: get grid shape
    '''
    grid_shape = np.array([(x_range[1] - x_range[0]) / cell_size[0],
                           (y_range[1] - y_range[0]) / cell_size[1]],
                          dtype=np.int)
    return grid_shape



def construct_bev(pcl, pcl_feature, x_range=(0,70), y_range=(-40, 40), cell_size=(0.25, 0.25)):
    '''
    :param pcl_feature: which point cloud channel to encode.
    :return:
    '''
    pcl[:,0] -= x_range[0]
    pcl[:,1] -= y_range[0]

    range_mask = mask_out_of_range_coors(pcl, x_range=x_range, y_range=y_range)

    pcl = pcl[range_mask]
    pcl_feature = pcl_feature[range_mask]

    xy_pcl = calculate_pcl_xy_coordinates(pcl, cell_size=cell_size)

    sort_mask = pcl_feature.argsort() # for maximum value
    xy_pcl = xy_pcl[sort_mask]
    pcl_feature = pcl_feature[sort_mask]

    grid_shape = calculate_shape(x_range, y_range, cell_size)
    bev = np.zeros(grid_shape)

    bev[xy_pcl[:,0], xy_pcl[:,1]] = pcl_feature

    return bev

def normalize_bev(bev):
    bev = (bev - bev.min()) / (bev.max() - bev.min())
    return bev

# if __name__ == '__main__':
#     print(ego_position((-125, 1021), (-700,100), (1,1)))
