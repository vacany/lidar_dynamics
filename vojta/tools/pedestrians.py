import numpy as np
#from tools.ground_removal import *
from sklearn.cluster import DBSCAN

MAX_MOVEMENT_SPEED_OF_PERSON = 0.8  # meters / tenth of a second
MIN_MOVEMENT_SPEED_OF_PERSON = 0.03  # meters / tenth of a second
EPSILON = 0.3  # meters
MIN_SAMPLES_PER_FRAME = 10
MIN_WIDTH = 0.2  # meters
MIN_LENGTH = 0.2  # meters
MIN_HEIGHT = 0.4  # meters
MIN_HEIGHT_CLUSTER = 0.7
MAX_HEIGHT = 1.8  # meters
MAX_WIDTH = 2
MAX_LENGTH = 2


def get_clusters(first_frame, num_of_frames):
    pts, _ = get_synchronized_frames_without_ground(first_frame, num_of_frames)  # X,Y,Z,Time
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES_PER_FRAME * num_of_frames, ).fit(pts[:, :3])
    print(f"formed {clustering.labels_.max() + 1} clusters")
    return pts, clustering

"""
def find_pedestrians_old(first_frame, num_of_frames):
    pts, clustering = get_clusters(first_frame, num_of_frames)
    differences = []

    for cluster in range(0, clustering.labels_.max() + 1):
        centroids = {}
        mask = clustering.labels_ == cluster
        cluster_points = pts[mask]

        # TODO consider average width,height,length instead of extremes - outliers
        width = np.abs(np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0]))
        length = np.abs(np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1]))
        height = np.abs(np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2]))

        # if the whole clulster is smaller or taller than limit person measures, skip
        if width <= MIN_WIDTH or length <= MIN_LENGTH or height <= MIN_HEIGHT \
                or height >= MAX_HEIGHT:
            print(f"skipping cluster {cluster} because of it's size w {width}, l {length}, h {height}")
            differences.append([0, 0, 0]) # required to retain same number of differences as number of clusters
            continue

        valid = True
        message = ""
        for time in range(first_frame, first_frame + num_of_frames):
            mask_time = cluster_points[:, 3] == time
            time_points = cluster_points[mask_time]
            if time_points.shape[0] == 0:
                message += f"cluster {cluster}, time {time} skipping because no points; "
                continue

            # if at each time frame, the cluster is smaller or bigger or taller than limit person measures, skip
            width = np.abs(np.max(time_points[:, 0]) - np.min(time_points[:, 0]))
            length = np.abs(np.max(time_points[:, 1]) - np.min(time_points[:, 1]))
            height = np.abs(np.max(time_points[:, 2]) - np.min(time_points[:, 2]))

            if width <= MIN_WIDTH or length <= MIN_LENGTH or height <= MIN_HEIGHT \
                    or height >= MAX_HEIGHT or width >= MAX_WIDTH or length >= MAX_LENGTH:
                print(
                    f"skipping cluster {cluster} at time {time} because of it's size w {width}, l {length}, h {height}")
                differences.append([0, 0, 0]) # required to retain same number of differences as number of clusters
                valid = False
                break

            centroid = np.sum(time_points, axis=0)[:3] / time_points.shape[0]
            centroids[time] = centroid

        if not valid:
            # skipping this cluster
            continue

        if len(message) > 0:
            print(message + "\n")

        # compute pairwise difference e.g: centroid[1] - centroid[0], centroid[2] - centroid[1]...
        centroids_keys = list(centroids.keys())
        sum_of_differences = 0
        num_of_differences = 0
        valid = True
        for i in range(len(centroids) - 1):
            # only considering differences between consecutive time frames
            if valid and centroids_keys[i + 1] == centroids_keys[i] + 1:
                difference = centroids[centroids_keys[i + 1]] - centroids[centroids_keys[i]]
                norm_of_difference_xy = np.linalg.norm(difference[:2])
                if norm_of_difference_xy < MAX_MOVEMENT_SPEED_OF_PERSON:
                    sum_of_differences += difference
                    num_of_differences += 1
                else:
                    valid = False
                    print(f"cluster {cluster} invalid because norm of difference at "
                          f"time {centroids_keys[i + 1]} and {centroids_keys[i]} is {norm_of_difference_xy} ")
                    break

        if valid:
            average_difference = sum_of_differences / num_of_differences
            # print(f"cluster {cluster} average difference {sum_of_differences / num_of_differences}")
            differences.append(average_difference)
        else:
            differences.append([0, 0, 0]) # required to retain same number of differences as number of clusters

    differences = np.array(differences)
    norm_of_differences = np.linalg.norm(differences[:, 0:2], axis=1)
    dynamic_clusters = np.argwhere((norm_of_differences >= MIN_MOVEMENT_SPEED_OF_PERSON)
                                   & (norm_of_differences <= MAX_MOVEMENT_SPEED_OF_PERSON))
    return pts, clustering.labels_, dynamic_clusters, differences
"""

def find_pedestrians(pts, times):
    """
    :param pts: n by 3 (or more) numpy array of XYZ coords
    :param times: n by 1 numpy array of times
    :return: mask of moving pedestrians, clustering (result from dbscan), labels
        of dynamic clusters, average differences of each cluster, centroids
    """
    time_values = np.unique(times)
    num_of_frames = time_values.shape[0]
    pts = np.concatenate((pts[:,:3], times), axis=1)
    clustering = DBSCAN(eps=0.4, min_samples=3 * num_of_frames, ).fit(pts[:, :3])
    print(f"formed {clustering.labels_.max() + 1} clusters")
    differences = []
    centroids_final = {}


    for cluster in range(0, clustering.labels_.max() + 1):
        centroids = {}
        mask = clustering.labels_ == cluster
        cluster_points = pts[mask]

        # TODO consider average width,height,length instead of extremes - outliers
        width = np.abs(np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0]))
        length = np.abs(np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1]))
        height = np.abs(np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2]))

        # if the whole clulster is smaller or taller than limit person measures, skip

        if width <= MIN_WIDTH or length <= MIN_LENGTH or height <= MIN_HEIGHT_CLUSTER \
                or height >= MAX_HEIGHT:
            print(f"skipping cluster {cluster} because of it's size w {width}, l {length}, h {height}")
            differences.append([0, 0, 0]) # required to retain same number of differences as number of clusters
            continue

        valid = True
        message = ""
        for time in range(num_of_frames):
            mask_time = cluster_points[:, 3] == time_values[time]
            time_points = cluster_points[mask_time]
            if time_points.shape[0] == 0:
                message += f"cluster {cluster}, time {time} skipping because no points; "
                continue

            # if at each time frame, the cluster is smaller or bigger or taller than limit person measures, skip
            width = np.abs(np.max(time_points[:, 0]) - np.min(time_points[:, 0]))
            length = np.abs(np.max(time_points[:, 1]) - np.min(time_points[:, 1]))
            height = np.abs(np.max(time_points[:, 2]) - np.min(time_points[:, 2]))

            #if (width <= MIN_WIDTH and length <= MIN_LENGTH) or height <= MIN_HEIGHT \
             #       or height >= MAX_HEIGHT or width >= MAX_WIDTH or length >= MAX_LENGTH:
            if height >= MAX_HEIGHT or max(width, length) >= MAX_WIDTH:
                print(
                    f"skipping cluster {cluster} at time {time} because of it's size w {width}, l {length}, h {height}")
                differences.append([0, 0, 0]) # required to retain same number of differences as number of clusters
                valid = False
                break

            centroid = np.sum(time_points, axis=0)[:3] / time_points.shape[0]
            centroids[time] = centroid

        if not valid:
            # skipping this cluster
            continue

        if len(message) > 0:
            print(message + "\n")

        # compute pairwise difference e.g: centroid[1] - centroid[0], centroid[2] - centroid[1]...
        centroids_keys = list(centroids.keys())
        sum_of_differences = 0
        num_of_differences = 0
        valid = True
        for i in range(len(centroids) - 1):
            # only considering differences between consecutive time frames
            if valid and centroids_keys[i + 1] == centroids_keys[i] + 1:
                difference = centroids[centroids_keys[i + 1]] - centroids[centroids_keys[i]]
                norm_of_difference_xy = np.linalg.norm(difference[:2])
                if norm_of_difference_xy < MAX_MOVEMENT_SPEED_OF_PERSON:
                    sum_of_differences += difference
                    num_of_differences += 1
                else:
                    valid = False
                    print(f"cluster {cluster} invalid because norm of difference at "
                          f"time {centroids_keys[i + 1]} and {centroids_keys[i]} is {norm_of_difference_xy} ")
                    break

        if valid and num_of_differences > 0:
            average_difference = sum_of_differences / num_of_differences
            # print(f"cluster {cluster} average difference {sum_of_differences / num_of_differences}")
            differences.append(average_difference)
            centroids_final[cluster] = centroids
        else:
            differences.append([0, 0, 0]) # required to retain same number of differences as number of clusters

    differences = np.array(differences)
    norm_of_differences = np.linalg.norm(differences[:, 0:2], axis=1)
    dynamic_clusters = np.argwhere((norm_of_differences >= MIN_MOVEMENT_SPEED_OF_PERSON)
                                   & (norm_of_differences <= MAX_MOVEMENT_SPEED_OF_PERSON))

    dynamic_mask = np.array([False * clustering.labels_.shape[0]])
    for dyn_cluster in dynamic_clusters:
        dynamic_mask = dynamic_mask | (clustering.labels_ == dyn_cluster)

    dynamic_mask = dynamic_mask.astype(bool)
    return dynamic_mask,clustering, dynamic_clusters, differences, centroids_final
