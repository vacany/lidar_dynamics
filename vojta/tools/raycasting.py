import numpy as np
from scipy.spatial import KDTree
from mayavi import mlab
from sklearn.cluster import DBSCAN


class RaycastPredictor():
    '''
    Class for predicting dynamic objects based on raycasting
    Parameters:
        config (dictionary): configuration dict of constants
        Dataloader (class): must have methods get_frame_without_ground and get_synchronized_origin
        fig (mayavi.mlab figure): figure to plot to
        verbose (bool): flag for printing more output
    '''
    def __init__(self, config, Dataloader, figure=None, verbose=False):
        self.config = config
        self.Dataloader = Dataloader
        self.figure = figure
        self.verbose = verbose


    def predict(self, num_of_frame):
        pts_current, _ = self.Dataloader.get_frame_without_ground(num_of_frame)
        pts_future, _ = self.Dataloader.get_frame_without_ground(num_of_frame + self.config["NUM_OF_FRAMES_IN_FUTURE"])

        if pts_future is not None:
            origin_future = self.Dataloader.get_synchronized_origin(num_of_frame + self.config["NUM_OF_FRAMES_IN_FUTURE"])
        else:
             origin_future = None

        if num_of_frame - self.config['NUM_OF_FRAMES_IN_FUTURE'] >= 0:
            pts_past, _ = self.Dataloader.get_frame_without_ground(num_of_frame - self.config["NUM_OF_FRAMES_IN_FUTURE"])            
            origin_past = self.Dataloader.get_synchronized_origin(num_of_frame - self.config["NUM_OF_FRAMES_IN_FUTURE"])
        else:
            pts_past = None
            origin_past = None

        num_of_pts_including_ground = self.Dataloader.get_frame(num_of_frame)[0].shape[0]
        clustering = DBSCAN(eps=self.config['EPS'], min_samples=self.config['MIN_SAMPLES']).fit(pts_current[:,:3])
        dynamic_mask = self.find_dynamic_objects(pts_current=pts_current, pts_future=pts_future, pts_past=pts_past,
             origin_future=origin_future, origin_past=origin_past, clustering=clustering)

        static_mask = self.find_static_objects(pts_current=pts_current, pts_future=pts_future,
             pts_past=pts_past, clustering=clustering)
        #complete_mask = np.array([False] * num_of_pts_including_ground)
        #no_ground_mask = self.Dataloader.get_frame_without_ground_mask(num_of_frame)
        #complete_mask[np.argwhere(no_ground_mask)] = dynamic_mask

        complete_mask = np.zeros((num_of_pts_including_ground,), dtype=np.uint32) # zero means unlabeled
        
        no_ground_mask = self.Dataloader.get_frame_without_ground_mask(num_of_frame)
        complete_mask[~no_ground_mask] = 9 # label for static obj      
        complete_mask[np.where(no_ground_mask)[0][static_mask]] = 9       
        complete_mask[np.where(no_ground_mask)[0][dynamic_mask]] = 251 # label for moving obj 
        
        return complete_mask



    def send_raycast(self, pts, target, target_points_indices = [], origin=[0,0,0], radius=0.3):
        '''
        Parameters:
            pts (numpy Nx3 array): point cloud
            target (1x3 array): coordinates of destination where we want to end our casted ray
            target_points_indices (numpy Mx1 array): indices of target (if known) so that we can check whether we have hit it
            origin (1x3 array): coordinates of ego (lidar) to cast ray from
            radius (float): radius of ray at each point 
        Returns:
            array of points which collided with ray
        '''
        
        target = target[:3] # make sure there is no intensity and only work with xyz coords
        target = target - origin
        distance = np.linalg.norm(target)
        num_of_checks = int(np.ceil((distance/radius) * 0.8)) # constant multiplication to make the spheres overlap less
        if self.verbose:
            print(f"Number of checks for raycast is {num_of_checks}", end=', ')

        # remove points that are part of ego so that they do not interfere with ray casting
        tree = KDTree(pts[:,:3])
        ego_points = tree.query_ball_point(origin, self.config['EGO_DELETE_RADIUS'])
        ego_mask = np.ones(pts.shape[0], dtype=bool)
        ego_mask[ego_points] = False
        pts = pts[ego_mask]
        tree = KDTree(pts[:,:3]) # without ego points
        
        if self.figure is not None:
            mlab.points3d(origin[0],origin[1],origin[2], scale_factor = 2 * radius,
                color=(1,0,0),figure=self.figure) # origin
            mlab.points3d(target[0] + origin[0], target[1] + origin[1], target[2] + origin[2],
                scale_factor = 2 * radius, color=(0,1,0),figure=self.figure)

        for t in range(1,num_of_checks):
            x,y,z = ((target * t)/ num_of_checks) + origin
            points_within_raycast_sphere = tree.query_ball_point([x,y,z], radius)
            if self.figure is not None:
                mlab.points3d(x,y,z, scale_factor = 2 * radius,
                    color=(0.3,0.5,1),figure=self.figure)
            if len(points_within_raycast_sphere) > 0:
                if np.sum(target_points_indices == points_within_raycast_sphere) > 0:
                    print(f"ray has hit target in points {points_within_raycast_sphere}")
                    return None
                if self.verbose:
                    print('ray has not reached the target')
                return points_within_raycast_sphere
        if self.verbose:
            print("ray has reached the target without hitting anything")
        return None


    def eliminate_objects_by_size(self, pts, labels):
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

            if length > self.config['MAX_LENGTH'] or length < self.config['MIN_LENGTH']:
                if self.verbose:
                    print(f"skipping cluster {cluster} because of its length {length}")
                continue
            if width > self.config['MAX_WIDTH']:
                if self.verbose:
                    print(f"skipping cluster {cluster} because of its width {width}")
                continue
            if height > self.config['MAX_HEIGTH'] or height < self.config['MIN_HEIGTH']:
                if self.verbose:
                    print(f"skipping cluster {cluster} because of its height {height}")
                continue

            valid_clusters.append(cluster)
        return np.array(valid_clusters)
            
    
    def find_static_objects(self, pts_current, pts_future, pts_past, clustering):
        if pts_future is not None:
            tree_future = KDTree(pts_future)
        if pts_past is not None:
            tree_past = KDTree(pts_past)
        clusters = max(clustering.labels_)
        static_clusters = []
        for cluster in range(0, clusters + 1):
            cluster_mask = clustering.labels_ == cluster
            centroid = np.mean(pts_current[:,:3][cluster_mask], axis=0)

            if pts_future is not None:
                points_close_to_centroid_future = tree_future.query_ball_point(
                                                    centroid, self.config['RADIUS_EMPTY_SPACE_CHECK'])
                if len(points_close_to_centroid_future) == 0:
                    continue

            if pts_past is not None:
                points_close_to_centroid_past = tree_past.query_ball_point(
                                                    centroid, self.config['RADIUS_EMPTY_SPACE_CHECK'])
                if len(points_close_to_centroid_past) == 0:
                    continue
            if self.verbose:
                print(f"cluster {cluster} is static")
            static_clusters.append(cluster)

        static_mask = np.array([False] * pts_current.shape[0])
        for cluster in static_clusters:
            static_mask = static_mask | (clustering.labels_ == cluster)
        static_mask = static_mask.astype(bool).reshape(-1,)
        return static_mask

    def get_potentionaly_moving_clusters(self, pts_current, pts_next, clustering, valid_clusters):
        tree = KDTree(pts_next[:,:3])
        potentially_moving_clusters = []
        for cluster in valid_clusters:
            cluster_mask = clustering.labels_ == cluster
            centroid = np.mean(pts_current[:,:3][cluster_mask], axis=0)
            points_close_to_centroid = tree.query_ball_point(centroid, self.config['RADIUS_EMPTY_SPACE_CHECK'])
            
            if len(points_close_to_centroid) == 0:
                # no points around what was previously a centroid of an object, meaning the object
                # must have moved or it cannot be seen because of other objects bloking it
                potentially_moving_clusters.append(cluster)
        return potentially_moving_clusters


    def find_dynamic_objects(self, pts_current, pts_future, pts_past, origin_future, origin_past, clustering):
        '''
        Parameters:
            pts_current (numpy Nx3 array): pointcloud from time T
            pts_future (numpy Mx3 array): pointcloud from time T+n, n >=1
            origin (numpy 1x3 array): coordinates of synchronized origin (coords of lidar)
        Returns:
            mask (numpy Nx1 array): mask of bools describing dynamic points 
        '''
        # get clusters filtered by size parameters specified in config
        
        valid_clusters = self.eliminate_objects_by_size(pts_current, clustering.labels_)

        # get moving objects
        if pts_future is not None:
            potentially_moving_clusters_future = self.get_potentionaly_moving_clusters(pts_current, 
                                                pts_future, clustering, valid_clusters)
        if pts_past is not None:
            potentially_moving_clusters_past = self.get_potentionaly_moving_clusters(pts_current, 
                                                pts_past, clustering, valid_clusters)
        
        # use raycasting to check whether we can see the approximate are of suspicious centroids, if so then
        # we proclaim them dynamic, if not, we cannot decide if they are dynamic or not
       # look few frames into the future
        true_dynamic_objects = []

        if pts_future is not None:
            for moving_cluster in potentially_moving_clusters_future:
                if self.verbose:
                    print(f"ray casting to cluster {moving_cluster} to future", end=' : ')
                cluster_mask = clustering.labels_ == moving_cluster
                centroid = np.mean(pts_current[:,:3][cluster_mask], axis=0)    
                response = self.send_raycast(pts_future, target=centroid,
                                        origin=origin_future, radius=self.config['RADIUS_RAYCAST'])
                if response is None:
                    true_dynamic_objects.append(moving_cluster)
        
        # look few frames into the past
        if pts_past is not None:
            # avoid duplicates
            potentially_moving_clusters_past = [x for x in potentially_moving_clusters_past if x not in true_dynamic_objects ]

            for moving_cluster in potentially_moving_clusters_past:
                if self.verbose:
                    print(f"ray casting to cluster {moving_cluster} to past", end=' : ')
                cluster_mask = clustering.labels_ == moving_cluster
                centroid = np.mean(pts_current[:,:3][cluster_mask], axis=0)
                response = self.send_raycast(pts_past, target=centroid,
                                        origin=origin_past, radius=self.config['RADIUS_RAYCAST'])
                if response is None:
                    true_dynamic_objects.append(moving_cluster)
                


        if self.verbose:
            print(f"dynamic objects: {true_dynamic_objects}")
        dynamic_mask = np.array([False] * pts_current.shape[0])
        
        for cluster in true_dynamic_objects:
            dynamic_mask = dynamic_mask | (clustering.labels_ == cluster)
        dynamic_mask = dynamic_mask.astype(bool).reshape(-1,)
        
        return dynamic_mask
