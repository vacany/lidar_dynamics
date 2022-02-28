import numpy as np
from scipy.spatial import KDTree
from mayavi import mlab
from sklearn.cluster import DBSCAN

#MAX_WIDTH = 3
#MAX_LENGTH = 6
#MAX_HEIGTH = 2
#MIN_LENGTH = 1


def send_raycast(pts, target, target_points_indices = [], origin=[0,0,0], radius=0.3, fig=None):
    '''
    Parameters:
        pts (numpy Nx3 array): point cloud
        target (1x3 array): coordinates of destination where we want to end our casted ray
        radius (float): radius of ray at each point
        fig (mayavi.mlab figure): figure to plot to
    Returns:
        array of points which collided with ray
    '''
    tree = KDTree(pts[:,:3])
    target = target[:3] # make sure there is no intensity and only work with xyz coords
    target = target - origin
    distance = np.linalg.norm(target)
    num_of_checks = int(np.ceil((distance/radius) * 1.2)) # constant multiplication to make the spheres overlap more
    print(f"Number of checks for raycast is {num_of_checks}")
    
    if fig is not None:
        mlab.points3d(origin[0],origin[1],origin[2], scale_factor = radius,
             color=(1,0,0),figure=fig) # origin
        mlab.points3d(target[0] + origin[0], target[1] + origin[1], target[2] + origin[2],
            scale_factor = radius, color=(0,1,0),figure=fig)

    for t in range(1,num_of_checks):
        x,y,z = ((target * t)/ num_of_checks) + origin
        points_within_raycast_sphere = tree.query_ball_point([x,y,z], radius)
        if fig is not None:
            mlab.points3d(x,y,z, scale_factor = radius,
                color=(0.3,0.5,1),figure=fig)
        if len(points_within_raycast_sphere) > 0:
            if np.sum(target_points_indices == points_within_raycast_sphere) > 0:
                print(f"ray has hit target in points {points_within_raycast_sphere}")
                return None
            return points_within_raycast_sphere
    print("ray has reached the destination without hitting anything")
    return None


def eliminate_objects_by_size(pts, labels, config, verbose=False):
    '''
    Parameters:
        pts (numpy Nx3 array): point cloud
        labels (numpy Nx1 array): result of DBSCAN labels_
    Returns:
        numpy Bx1 array of clusters which are within size norms
    '''
    valid_clusters = []
    for cluster in range(0, labels.max()):
        mask = labels == cluster
        cluster_pts = pts[mask]

        bb_x_min =  cluster_pts[:,0].min()
        bb_x_max =  cluster_pts[:,0].max()
        bb_y_min =  cluster_pts[:,1].min()
        bb_y_max =  cluster_pts[:,1].max()
        bb_z_min =  cluster_pts[:,2].min()
        bb_z_max =  cluster_pts[:,2].max()

        width = min(abs(bb_x_max - bb_x_min), abs(bb_y_max - bb_y_min))
        length = max(abs(bb_x_max - bb_x_min), abs(bb_y_max - bb_y_min))
        height = abs(bb_z_max - bb_z_min)

        if length > config['MAX_LENGTH'] or length < config['MIN_LENGTH']:
            if verbose:
                print(f"skipping cluster {cluster} because of its length {length}")
            continue
        if width > config['MAX_WIDTH']:
            if verbose:
                print(f"skipping cluster {cluster} because of its width {width}")
            continue
        if height > config['MAX_HEIGTH']:
            if verbose:
                print(f"skipping cluster {cluster} because of its height {height}")
            continue
        valid_clusters.append(cluster)
    return np.array(valid_clusters)
            

def find_dynamic_objects(pts, pts2, origin, config, figure=None):
    '''
    Parameters:
        pts (numpy Nx3 array): pointcloud from time T
        pts2 (numpy Mx3 array): pointcloud from time T+n, n >=1
        origin (numpy 1x3 array): coordinates of synchronized origin (coords of lidar)
        config (dictionary): configuration dict of constants
        figure (mayavi.mlab figure): figure to plot to
    Returns:
        mask (numpy Nx1 array): mask of bools describing dynamic points 
    '''
    # get clusters filtered by size parameters specified in config
    clustering = DBSCAN(eps=config['EPS'], min_samples=config['MIN_SAMPLES']).fit(pts[:,:3])
    valid_clusters = eliminate_objects_by_size(pts, clustering.labels_, config, verbose=False)

    # get moving objects
    tree = KDTree(pts2[:,:3])
    potentially_moving_clusters = []
    for cluster in valid_clusters:
        cluster_mask = clustering.labels_ == cluster
        centroid = np.mean(pts[:,:3][cluster_mask], axis=0)
        points_close_to_centroid = tree.query_ball_point(centroid, config['RADIUS_EMPTY_SPACE_CHECK'])
        if len(points_close_to_centroid) == 0:
            # no points around what was previously a centroid of an object, meaning the object
            # must have moved or it cannot be seen because of other objects bloking it
            potentially_moving_clusters.append(cluster)
    
    # use raycasting to check whether we can see the approximate are of suspicious centroids, if so then
    # we proclaim them dynamic, if not, we cannot decide if they are dynamic or not
    true_dynamic_objects = []
    for moving_cluster in potentially_moving_clusters:
        cluster_mask = clustering.labels_ == moving_cluster
        centroid = np.mean(pts[:,:3][cluster_mask], axis=0)
        response = send_raycast(pts2, target=centroid,
                                origin=origin, radius=config['RADIUS_RAYCAST'], fig=figure)
        if response is None:
            true_dynamic_objects.append(moving_cluster)
    
    print(f"dynamic objects: {true_dynamic_objects}")
    dynamic_mask = np.array([False * pts.shape[0]])
    for cluster in true_dynamic_objects:
        dynamic_mask = dynamic_mask | (clustering.labels_ == cluster)
    dynamic_mask = dynamic_mask.astype(bool)
    return dynamic_mask
