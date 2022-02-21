import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from exps.utils import timeit

class Point_Cloud_Labeller():
    def __init__(self, config):
        self.config = config

    def update(self, batch):
        ''' make smoother? just visual problem, works anyway ... '''
        trans = batch['all_poses'][batch['frame']]
        local_pcl = batch['points'].copy()
        inten = local_pcl[:, 3].copy()

        local_pcl[:, 3] = 1

        local_pcl[:, :3] = (np.linalg.inv(trans) @ local_pcl[:, :4].T)[:3, :].T
        local_pcl[:, 3] = inten

        self.local_pcl = local_pcl
        self.global_pcl = batch['points']

        self.seg_mask = np.zeros(self.local_pcl[:, 0].shape, dtype=np.int) - 1
        self.cluster_mask = np.zeros(self.local_pcl[:, 0].shape, dtype=np.int) - 1
        self.instance_mask = np.zeros(self.local_pcl[:, 0].shape, dtype=np.int) - 1

    def get_reflective_surface(self):
        self.inten_mask = self.local_pcl[:,3] > self.config['inten_threshold']

    def split_reflective_by_height(self):
        inten_mask = self.get_features('inten_mask')
        height_mask = self.local_pcl[:,2] > self.config['Traffic_Sign']['min_height']
        self.traffic_sighs = inten_mask * height_mask

    def visual_testing(self):
        # from data.point_clouds.visual import render
        # render.Vis_function(dataset=dataset, function=remove_ground, config=self.config, offset=offset)
        import pptk
        v=pptk.viewer(self.local_pcl[:,:3], self.inten_mask)
        v.set(point_size=0.02)

    def store_features(self, mask, name):
        if not hasattr(self, name):
            setattr(self, name, mask)

    def get_features(self, name):
        mask = getattr(self, name)
        return mask

def split_car_and_DA(pcl, xy_pcl, da_mask, height_map, margin):
    '''
    :param height_map: Bev with heights colored from ego
    :param margin: height of the ego-car in some sense
    :return:
    '''
    z = pcl[:, 2]
    car_mask = (z > height_map[xy_pcl[:, 0], xy_pcl[:, 1]] + margin) * da_mask
    da_mask[car_mask] = 0

    return car_mask

def cluster_chosen_points(pcl, mask, min_dist=0.3, min_samples=3):
    '''
    :param pcl:
    :param mask: pcl mask which denotes the points to be clustered
    :return:
    '''
    cluster_model = DBSCAN(min_dist, min_samples=min_samples)

    cluster_model.fit(pcl[mask])
    clusters = cluster_model.labels_

    cluster_mask = np.zeros(pcl.shape[0], dtype=np.bool)
    indices = np.argwhere(mask==True)[:,0]
    cluster_mask[indices] = clusters

    return cluster_mask

@timeit
def distance_from_points(pcl, points, max_radius=5):
    '''
    :param points: xyz special points, which define area(radius) of interest
    :param max_radius:
    :return:
    '''
    mask = np.zeros(pcl.shape[0], dtype=np.bool)
    for point in points:
        coors = pcl[:,:3] - point[None,:3]
        distance = np.sqrt(np.sum(coors ** 2, axis=1))
        mask += distance < max_radius

    dist_mask = mask > 0

    return dist_mask


def get_unique_clusters(pcl, potential_mask, true_mask, min_dist=0.3, min_samples=10):
    '''
    :param potential_mask: Mask of valid points to be specific class
    :param true_mask: Mask of points, which are gauranteed to be right, i.e. licence plates
    :param dist: Clustering constant
    :param min_samples: Clustering constant
    :return: Full mask with clusters of the desired class
    '''
    model = DBSCAN(min_dist, min_samples=min_samples)
    label_mask = np.zeros(pcl.shape[0])

    model.fit(pcl[potential_mask])
    all_labels = model.labels_
    right_labels = true_mask[potential_mask]

    tmp_clusters = np.unique(all_labels[right_labels])
    for i in range(len(all_labels)):
        if all_labels[i] in tmp_clusters:
            all_labels[i] = np.argwhere(all_labels[i] == tmp_clusters)

        else:
            all_labels[i] = 0

    indices = np.argwhere(potential_mask == True)[:, 0]
    label_mask[indices] = all_labels

    return label_mask, all_labels


def get_licence_plates(pcl, inten_threshold=0.97, margin=-0.6):
    '''
    :param inten_threshold: Distinguish between licence plates, traffic signs and background
    :return:
    '''
    licence_mask = (pcl[:,3] > inten_threshold) * (pcl[:,2] < margin) #TODO This height should be systematic
    # signs_mask = (pcl[:,3] > inten_threshold) * (pcl[:,2] > -0.6)

    inten_mask = licence_mask

    return inten_mask

def calculate_height_variance(inten_pcl, clusters, z_var_thres=0.2):

    seg_mask = np.zeros(inten_pcl.shape[0], dtype=np.float)

    # Noise from clustering method
    seg_mask[clusters == -1] = 0
    for i in np.unique(clusters):
        mask = clusters == i

        z_var = inten_pcl[mask,2].max() - inten_pcl[mask,2].min()

        if z_var >= z_var_thres:
            label = 2
        elif z_var < z_var_thres:
            label = 1
        seg_mask[mask] = label

    return seg_mask

def get_farthest_points(points):

    # Find a convex hull in O(N log N)
    hull = ConvexHull(points)

    # Extract the points forming the hull
    hullpoints = points[hull.vertices, :]

    # Naive way of finding the best pair in O(H^2) time if H is number of points on
    # hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')

    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    farthest_points = ([hullpoints[bestpair[0]], hullpoints[bestpair[1]]])

    return np.linalg.norm(farthest_points[0] - farthest_points[1], ord=2) # ord == euclidian

def label_high_static_objects(pcl, config):
    ''' not refactored'''
    height_mask = pcl[:,2] > 0 + config['add_ground_height']

    depth = np.linalg.norm(pcl[:,:3], 2, axis=1)
    pitch = np.arcsin((pcl[:,2]) / depth)

    # surrounding area


#TODO height plane is changing because frames are globally on different spots!
    # mask = (pitch > 0.01) & (depth < 40)
    # seg[mask] = 0

def filter_building(pcl, cluster_mask, max_size=8):
    ''' cannot do it like this because car-building-da cluster?'''

    instance_seg = np.zeros(cluster_mask.shape)

    for cluster in np.unique(cluster_mask):
        # cluster = 0 is background
        if cluster == 0:
            continue

        distance = get_farthest_points(pcl[cluster_mask == cluster][:, :3])  # can be one point?
        #print(distance) add if verbose?
        if distance > max_size:
            instance_seg[cluster_mask == cluster] = -1

        else:
            instance_seg[cluster_mask == cluster] = cluster

    return instance_seg

if __name__ == "__main__":
    from exps.rw import load_yaml
    from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
    import os

    config = load_yaml(os.path.dirname(os.path.abspath(__file__).split('.')[0]) + '/config_values.yaml')
    dataset = SemKittiDataset()

    labeler = Point_Cloud_Labeller(config)

    # def function_(labeler, config):


    # from data.point_clouds.visual import render
    # render.Vis_function(dataset=dataset, function=remove_ground, config=self.config, offset=offset)
