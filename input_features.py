import numpy as np

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
# from shapely.geometry import Polygon, Point

''' GEOMETRICAL FEATURES '''
def get_indices_by_mask(mask):
    return np.argwhere(mask==True)[:,0]

def calibrate_height(local_pcl, sensor_height=1.8):
    local_pcl[:,2] = local_pcl[:,2] + sensor_height
    return local_pcl

def mask_by_distance(pcl, max_distance=30):
    distance = np.sqrt(pcl[:,:3] ** 2).sum(1)
    mask = distance <= max_distance
    return mask

def remove_by_distance(pcl, max_distance=30, return_mask=False):
    mask = mask_by_distance(pcl, max_distance)
    if not return_mask:
        return pcl[mask]
    else:
        return pcl[mask], mask

def mask_ground(local_pcl, ground_removal_noise=0.2):
    ground_mask = local_pcl[:, 2] > ground_removal_noise
    return ground_mask

def remove_ground(local_pcl, ground_removal_noise=0.2, return_mask=False):
    mask = mask_ground(local_pcl, ground_removal_noise)
    if not return_mask:
        return local_pcl[mask]
    else:
        return local_pcl[mask], mask

def remove_ground_and_distance(local_pcl, ground_removal_noise=0.2, max_distance=30, return_mask=False):
    ground_mask = mask_ground(
            local_pcl,
            ground_removal_noise=ground_removal_noise
    )

    dist_mask = mask_by_distance(
            local_pcl,
            max_distance=max_distance
    )

    all_mask = (dist_mask & ground_mask)

    return all_mask

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
    return farthest_points

def get_max_size(points):
    if len(points) == 2:
        return np.sqrt(((points[1] - points[0]) ** 2).sum())

    else:
        farthest_points = get_farthest_points(points)

    return np.linalg.norm(farthest_points[0] - farthest_points[1], ord=2) # ord == euclidian

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

""" Intensity """

def get_reflective_surface(local_pcl, inten_thres=0.95):
    inten_mask = local_pcl[:, 3] > inten_thres
    return inten_mask

def get_traffic_sign(local_pcl, inten_mask, min_height=1.1):
    height_mask = local_pcl[:, 2] > min_height
    traffic_sighs = inten_mask & height_mask
    return traffic_sighs

def get_license_plate(local_pcl, inten_mask, max_height=0.4):
    license_plates = inten_mask & (local_pcl[:,2] < max_height)
    return license_plates

def get_vehicle_plates(local_pcl, license_plates, eps=0.5, min_samples=2):
    plate_clusters = np.zeros(local_pcl.shape[0], dtype=np.int)

    if not license_plates.any():
        return plate_clusters

    plate_model = DBSCAN(eps=eps, min_samples=min_samples)
    plate_model.fit(local_pcl[license_plates, :3])

    indices = get_indices_by_mask(license_plates)
    plate_clusters[indices] = plate_model.labels_

    return plate_clusters


def eliminate_oversized_plates(local_pcl, plate_clusters, max_plate_z_var=0.3):
    # for next time, you can split it to features
    ''' Eliminate one with high z variance '''
    if np.sum(plate_clusters) == 0:
        return plate_clusters

    for id in np.unique(plate_clusters):
        # -1 is for invalid clusters
        if id == -1: continue

        plate_pcl = local_pcl[plate_clusters == id, :3]
        z_var = np.abs(plate_pcl[:, 2].max() - plate_pcl[:, 2].min())

        if z_var > max_plate_z_var:
            # remove cluster from consideration
            plate_clusters[plate_clusters == id] = -1
            # TODO check this properly?

    return plate_clusters

''' Temporal '''


def mask_frames_by_time(pcl, from_time, till_time):

    time_mask = (pcl[:,4] >= from_time) & (pcl[:,4] <= till_time)
    return time_mask

#TODO continue - functions, then wrapper - for protocol as well
#TODO lanes, Do I see the ground after moving object?


if __name__ == '__main__':
    from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
    dataset = SemKittiDataset(prev=20, sequence=[18], every_th=3)
    batch = dataset.get_multi_frame(20)

    pcl = batch['points'][:]

    import pptk


    # Geometry
    pcl = calibrate_height(pcl)
    pcl = pcl[remove_ground_and_distance(pcl)]

    # Intensity
    inten_mask = get_reflective_surface(pcl)
    plate_mask = get_license_plate(pcl, inten_mask)

    clusters = get_vehicle_plates(pcl, inten_mask)

    clusters = eliminate_oversized_plates(pcl, clusters)

    v = pptk.viewer(pcl[:, :3], clusters)
    v.set(point_size=0.01)



