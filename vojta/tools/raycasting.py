import numpy as np
from scipy.spatial import KDTree
from mayavi import mlab

MAX_WIDTH = 3
MAX_LENGTH = 6
MAX_HEIGTH = 2
MIN_LENGTH = 1


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


def eliminate_objects_by_size(pts, labels):
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

        if length > MAX_LENGTH or length < MIN_LENGTH:
            print(f"skipping cluster {cluster} because of its length {length}")
            continue
        if width > MAX_WIDTH:
            print(f"skipping cluster {cluster} because of its width {width}")
            continue
        if height > MAX_HEIGTH:
            print(f"skipping cluster {cluster} because of its height {height}")
            continue
        valid_clusters.append(cluster)
    return np.array(valid_clusters)
            