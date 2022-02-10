import numpy as np
# from tools.ground_removal import *
from sklearn.cluster import DBSCAN

# kitti parameters

MAX_MOVEMENT_SPEED_OF_PERSON = 0.8  # meters / tenth of a second
MIN_MOVEMENT_SPEED_OF_PERSON = 0.06  # meters / tenth of a second
EPSILON = 0.3  # meters
MIN_SAMPLES_PER_FRAME = 7
MIN_WIDTH = 0.2  # meters
MIN_LENGTH = 0.2  # meters
MIN_HEIGHT = 0.4  # meters
MIN_HEIGHT_CLUSTER = 0.5 # meters
MAX_HEIGHT = 1.8  # meters
MAX_WIDTH = 2 # meters
MAX_LENGTH = 2 # meters

PCL_HEIGTH_UPPER_BOUND = 1 # meters
PCL_HEIGTH_LOWER_BOUND = -1 # meters

GROUND_HEIGHT_OFFSET = 0.1 # meters
MINIMAL_TIME_WINDOW_FOR_GROUND_CHECKING = 4 # frames
MINIMAL_NUMBER_OF_GROUND_POINTS = 2 
MAXIMAL_DISTANCE_FROM_GROUND = 0.4

"""
# Nuscenes parameters

MAX_MOVEMENT_SPEED_OF_PERSON = 0.8  # meters / tenth of a second
MIN_MOVEMENT_SPEED_OF_PERSON = 0.03  # meters / tenth of a second
EPSILON = 0.4  # meters
MIN_SAMPLES_PER_FRAME = 3
MIN_WIDTH = 0.2  # meters
MIN_LENGTH = 0.2  # meters
MIN_HEIGHT = 0.4  # meters
MIN_HEIGHT_CLUSTER = 0.7
MAX_HEIGHT = 1.8  # meters
MAX_WIDTH = 2
MAX_LENGTH = 2

PCL_HEIGTH_UPPER_BOUND = 1
PCL_HEIGTH_LOWER_BOUND = -1
"""


def find_pedestrians(pts_uncropped, times_uncropped, pts_cropped, times_cropped):
    """
    :param pts_uncropped: N by 3 numpy array of XYZ coords
    :param times_uncropped: N by 1 numpy array of times (must be shape of N by 1!) 
    :param pts_cropped: M by 3 (M <= N) numpy array of XYZ cropped coords 
    :param times_cropped: M by 1 array of cropped times
    :return: 5 values ->  mask of moving pedestrians (dimension of M), clustering (result from dbscan), labels
        of dynamic clusters, average differences of each cluster, centroids
    """
    
    time_values = np.unique(times_uncropped)
    time_values.sort()
    num_of_frames = time_values.shape[0]
    #pts = np.concatenate((pts[:, :3], times), axis=1)
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES_PER_FRAME * num_of_frames, ).fit(pts_cropped[:, :3])
    print(f"formed {clustering.labels_.max() + 1} clusters")
    differences = []
    centroids_final = {}

    for cluster in range(0, clustering.labels_.max() + 1):
        centroids = {}
        mask = clustering.labels_ == cluster
        cluster_points = pts_cropped[mask]
        cluster_times = times_cropped[mask]

        # TODO consider average width,height,length instead of extremes - outliers
        width = np.abs(np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0]))
        length = np.abs(np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1]))
        height = np.abs(np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2]))

        # if the whole clulster is smaller or taller than limit person measures, skip

        if width <= MIN_WIDTH or length <= MIN_LENGTH or height <= MIN_HEIGHT_CLUSTER \
                or height >= MAX_HEIGHT:
            print(f"skipping cluster {cluster} because of it's size w {width}, l {length}, h {height}")
            differences.append([0, 0, 0])  # required to retain same number of differences as number of clusters
            continue

        valid = True
        message = ""
        for time in range(num_of_frames):
            mask_time = cluster_times == time_values[time]
            mask_time = mask_time.reshape(-1,)
            
            time_points = cluster_points[mask_time]
            if time_points.shape[0] == 0:
                message += f"cluster {cluster}, time {time} skipping because no points; "
                continue

            # if at each time frame, the cluster is smaller or bigger or taller than limit person measures, skip
            width = np.abs(np.max(time_points[:, 0]) - np.min(time_points[:, 0]))
            length = np.abs(np.max(time_points[:, 1]) - np.min(time_points[:, 1]))
            height = np.abs(np.max(time_points[:, 2]) - np.min(time_points[:, 2]))

            # if (width <= MIN_WIDTH and length <= MIN_LENGTH) or height <= MIN_HEIGHT \
            #       or height >= MAX_HEIGHT or width >= MAX_WIDTH or length >= MAX_LENGTH:
            if height >= MAX_HEIGHT or max(width, length) >= MAX_WIDTH:
                print(
                    f"skipping cluster {cluster} at time {time} because of it's size w {width}, l {length}, h {height}")
                differences.append([0, 0, 0])  # required to retain same number of differences as number of clusters
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
            differences.append([0, 0, 0])  # required to retain same number of differences as number of clusters

    differences = np.array(differences)
    norm_of_differences = np.linalg.norm(differences[:, 0:2], axis=1)
    dynamic_clusters = np.argwhere((norm_of_differences >= MIN_MOVEMENT_SPEED_OF_PERSON)
                                   & (norm_of_differences <= MAX_MOVEMENT_SPEED_OF_PERSON))

    dynamic_clusters = dynamic_clusters.reshape(-1,)
  

    """ 
    ---------------------------------------------------------------------------------------------
    check whether the ground below the pedestrian is still visible after some frames if it is not
    then we cannot say with certainity if the dynamic object is pedestrian    
    """
    
    
    print("\n---------second wave of filtering--------------\n")
    filtered_dynamic_clusters = []
    print(f"considering {dynamic_clusters}")
    for cluster in dynamic_clusters:    
        single_cluster_mask = clustering.labels_ == cluster
        
        # find first time the object is visible
        first_time = -1
        for time in time_values:
            if np.sum(times_cropped[single_cluster_mask] == time) > 1: # to make sure there are atleast two points
                first_time = time
                break 
        print(f"cluster {cluster} first time = {first_time}")       
        if (time_values[-1] - first_time) < MINIMAL_TIME_WINDOW_FOR_GROUND_CHECKING:
            print(f"skipping cluster {cluster} because it is not observed in enough frames,"
                  f"first time of observation is {first_time}")
            continue # skipping this cluster, declaring it static
            
        time_mask = times_cropped[single_cluster_mask] == first_time
        time_mask = time_mask.reshape(-1,)
        
        # bounding box
        bb_x_min =  pts_cropped[:,0][single_cluster_mask][time_mask].min()
        bb_x_max =  pts_cropped[:,0][single_cluster_mask][time_mask].max()
        bb_y_min =  pts_cropped[:,1][single_cluster_mask][time_mask].min()
        bb_y_max =  pts_cropped[:,1][single_cluster_mask][time_mask].max()
        bb_z_min = pts_cropped[:,2][single_cluster_mask][time_mask].min()
        
        bb_mask = (pts_uncropped[:,0] < bb_x_max) & (pts_uncropped[:,0] > bb_x_min) \
            & (pts_uncropped[:,1] < bb_y_max) & (pts_uncropped[:,1] > bb_y_min)
        
        if np.sum(bb_mask) == 0:
            print("zero mask points")
            continue
        ground = pts_uncropped[:,2][bb_mask].min()
        ground = max(ground, bb_z_min - MAXIMAL_DISTANCE_FROM_GROUND) # make sure there is no more than MDFG meters 
        # between end of cropped feet and lowest ground point, eliminating objects which are not on sidewalk/ground and errors due to glass/mirrors
        
        # TODO use knowledge of where the ground approximately is and eliminate objects too high

        ground_mask = bb_mask & (pts_uncropped[:,2] < (ground + GROUND_HEIGHT_OFFSET))
        
        valid = True
        observed_times_of_ground, num_of_points_at_time = np.unique(times_uncropped[ground_mask], return_counts=True)
        
        
        print(f"cluster {cluster} ground observed at times {observed_times_of_ground}")
        if observed_times_of_ground[num_of_points_at_time > MINIMAL_NUMBER_OF_GROUND_POINTS].shape[0] < MINIMAL_TIME_WINDOW_FOR_GROUND_CHECKING:
            print(f"skipping cluster {cluster} because ground is only observed at times {observed_times_of_ground} at frequencies {num_of_points_at_time}")
            continue
        else:
            filtered_dynamic_clusters.append(cluster)
            
    """
    ---------------------------------------------------------------------------------------------
    """
    
    
    #filtered_dynamic_clusters = dynamic_clusters
    
    dynamic_mask = np.array([False * clustering.labels_.shape[0]])
    for dyn_cluster in filtered_dynamic_clusters:
        dynamic_mask = dynamic_mask | (clustering.labels_ == dyn_cluster)

    dynamic_mask = dynamic_mask.astype(bool)
    return dynamic_mask, clustering, filtered_dynamic_clusters, differences, centroids_final


def find_pedestrians_old(pts_, times_):
    """
    :param pts: N by 3 (or more) numpy array of XYZ coords
    :param times: N by 1 numpy array of times (must be shape of N by 1!) 
    :return: 6 values -> cropped pts, mask of moving pedestrians (with same dimensions
        as copped pts), clustering (result from dbscan), labels
        of dynamic clusters, average differences of each cluster, centroids
    """

    
    pts = pts_.copy()
    times = times_.copy()
    mask = (pts[:,2] > PCL_HEIGTH_LOWER_BOUND) & (pts[:,2] < PCL_HEIGTH_UPPER_BOUND)
    pts = pts[mask]
    times = times[mask]

    time_values = np.unique(times)
    time_values.sort()
    num_of_frames = time_values.shape[0]
    pts = np.concatenate((pts[:, :3], times), axis=1)
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES_PER_FRAME * num_of_frames, ).fit(pts[:, :3])
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
            differences.append([0, 0, 0])  # required to retain same number of differences as number of clusters
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

            # if (width <= MIN_WIDTH and length <= MIN_LENGTH) or height <= MIN_HEIGHT \
            #       or height >= MAX_HEIGHT or width >= MAX_WIDTH or length >= MAX_LENGTH:
            if height >= MAX_HEIGHT or max(width, length) >= MAX_WIDTH:
                print(
                    f"skipping cluster {cluster} at time {time} because of it's size w {width}, l {length}, h {height}")
                differences.append([0, 0, 0])  # required to retain same number of differences as number of clusters
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
            differences.append([0, 0, 0])  # required to retain same number of differences as number of clusters

    differences = np.array(differences)
    norm_of_differences = np.linalg.norm(differences[:, 0:2], axis=1)
    dynamic_clusters = np.argwhere((norm_of_differences >= MIN_MOVEMENT_SPEED_OF_PERSON)
                                   & (norm_of_differences <= MAX_MOVEMENT_SPEED_OF_PERSON))

    dynamic_clusters = dynamic_clusters.reshape(-1,)
  

    """ 
    ---------------------------------------------------------------------------------------------
    check whether the ground below the pedestrian is still visible after some frames if it is not
    then we cannot say with certainity if the dynamic object is pedestrian    
    """
    
    
    print("\n---------second wave of filtering--------------\n")
    filtered_dynamic_clusters = []
    print(f"considering {dynamic_clusters}")
    for cluster in dynamic_clusters:    
        single_cluster_mask = clustering.labels_ == cluster
        
        # find first time the object is visible
        first_time = -1
        for time in time_values:
            if np.sum(pts[:,3][single_cluster_mask] == time) > 1: # to make sure there are atleast two points
                first_time = time
                break 
        print(f"cluster {cluster} first time = {first_time}")       
        if (time_values[-1] - first_time) < MINIMAL_TIME_WINDOW_FOR_GROUND_CHECKING:
            print(f"skipping cluster {cluster} because it is not observed in enough frames,"
                  f"first time of observation is {first_time}")
            continue # skipping this cluster, declaring it static
            
        time_mask = pts[:,3][single_cluster_mask] == first_time
        
        # bounding box
        bb_x_min =  pts[:,0][single_cluster_mask][time_mask].min()
        bb_x_max =  pts[:,0][single_cluster_mask][time_mask].max()
        bb_y_min =  pts[:,1][single_cluster_mask][time_mask].min()
        bb_y_max =  pts[:,1][single_cluster_mask][time_mask].max()
        bb_z_min = pts[:,2][single_cluster_mask][time_mask].min()
        
        bb_mask = (pts_[:,0] < bb_x_max) & (pts_[:,0] > bb_x_min) \
            & (pts_[:,1] < bb_y_max) & (pts_[:,1] > bb_y_min)
        
        if np.sum(bb_mask) == 0:
            print("zero mask points")
            continue
        ground = pts_[:,2][bb_mask].min()
        ground_mask = bb_mask & (pts_[:,2] < (ground + GROUND_HEIGHT_OFFSET))
        
        valid = True
        observed_times_of_ground, num_of_points_at_time = np.unique(times_[ground_mask], return_counts=True)
        
        
        print(f"cluster {cluster} ground observed at times {observed_times_of_ground}")
        if observed_times_of_ground[num_of_points_at_time > MINIMAL_NUMBER_OF_GROUND_POINTS].shape[0] < MINIMAL_TIME_WINDOW_FOR_GROUND_CHECKING:
            print(f"skipping cluster {cluster} because ground is only observed at times {observed_times_of_ground} at frequencies {num_of_points_at_time}")
            continue
        else:
            filtered_dynamic_clusters.append(cluster)
            
    
    """ --------
            for t in observed_times_of_ground:
                if np.sum(times_[ground_mask] == t) < MINIMAL_NUMBER_OF_GROUND_POINTS:
                    print(f"skipping cluster {cluster} because at time {t} there are too few ground points")
                    valid = False
                    break
            if valid:
                filtered_dynamic_clusters.append(cluster)
   """
    
    
    """
    ---------------------------------------------------------------------------------------------
    """
    
    
    #filtered_dynamic_clusters = dynamic_clusters
    
    dynamic_mask = np.array([False * clustering.labels_.shape[0]])
    for dyn_cluster in filtered_dynamic_clusters:
        dynamic_mask = dynamic_mask | (clustering.labels_ == dyn_cluster)

    dynamic_mask = dynamic_mask.astype(bool)
    return pts, dynamic_mask, clustering, filtered_dynamic_clusters, differences, centroids_final
