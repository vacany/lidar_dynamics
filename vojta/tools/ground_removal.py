import numpy as np
import os
import random
from scipy.spatial import KDTree
#from ground_removal_kitti import remove_ground

class Ground_removal:
    def __init__(self, sequence, path=None): 
        if path is None:
            self.path = "../../semantic_kitti_data/sequences/" + sequence + "/"
        else:
            self.path=path
        self.poses = np.loadtxt(self.path + "poses.txt")
        self.poses = self.poses.reshape(-1, 3, 4)


    def read_calibration_file(self, ):
        file = os.path.join(self.path,"calib.txt")
        calibration_tr = ""
        with open(file) as f:
            for ln in f:
                if ln.startswith("Tr: "):
                    calibration_tr = ln[4:-1]
        
        calibration_tr = np.array(calibration_tr.split(' '), dtype=np.float64)
        calibration_tr = calibration_tr.reshape(3,4)
        calibration_tr = np.vstack((calibration_tr, np.array([0,0,0,1])))
        
        return calibration_tr

    def transform_mat(self, _pts, pose):
        """
        multiply matrix of 3d points by transformation matrix, which will result in original 3d points
        being synchronized across multiple sequences, meaning a rigid object will stay in place even though
        the LIDAR is moving
        """
        
        tr = self.read_calibration_file()
        tr_inv = np.linalg.inv(tr)
        pose = np.matmul(tr_inv, np.matmul(pose, tr))
        n, _ = _pts.shape
        x = np.hstack((_pts, np.ones((n, 1))))
        x = np.matmul(pose, x.transpose()).transpose()
        return x[:, 0:3]


    def get_labels(self, number):
        file_name = "0" * int(6 - (len(str(number)))) + str(number)
        label_path = self.path + "labels/" + file_name + ".label"
        if not os.path.exists(label_path):
            print(f"Error, path {label_path} does not exists")
            return None, None
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))

        pts_path = self.path + "velodyne/" + file_name + ".bin"
        if not os.path.exists(pts_path):
            print(f"Error, path {pts_path} does not exists")
            return None, None
        points = np.fromfile(pts_path, dtype=np.float32).reshape(-1,4)

        if label.shape[0] == points.shape[0]:
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16    # instance id in upper half
        else:
            print("Points shape: ", points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert((sem_label + (inst_label << 16) == label).all())
        return sem_label


    def get_synchronized_origin(self, number):
        if number < 0:
            print(f"Error, number {number} is invalid for origin")
            return None
        pose = self.poses[number]
        pose = np.vstack((pose, [0, 0, 0, 1]))
        origin = np.array([0,0,0]).reshape(1,-1)
        origin = self.transform_mat(origin, pose).flatten()
        return origin


    def get_frame(self, number):
        """
        Loads and synchronizes PCL

            Parameters:
                number (int): number of PCL sequence to be loaded

            Returns:
                pts (numpy.array): N by 3 array of synchronized PCL
                intensities (numpy.array): N by 1 array of intensities of points in PCL
        """
        file_name = "0" * int(6 - (len(str(number)))) + str(number)
        file_path = self.path + "velodyne/" + file_name + ".bin"
        if not os.path.exists(file_path):
            print(f"Error, path {file_path} does not exists")
            return None, None
        scan = np.fromfile(file_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        intensities = scan[:, 3]
        pts = scan[:, 0:3]

        pose = self.poses[number]
        pose = np.vstack((pose, [0, 0, 0, 1]))
        pts = self.transform_mat(pts, pose).reshape(-1, 3)

        return pts, intensities


    def get_frame_unsynchronized(self, number):
        """
        Loads PCL without synchronization

            Parameters:
                number (int): number of PCL sequence to be loaded

            Returns:
                pts (numpy.array): N by 3 array of  PCL
                intensities (numpy.array): N by 1 array of intensities of points in PCL
        """
        file_name = "0" * int(6 - (len(str(number)))) + str(number)
        file_path = self.path + "velodyne/" + file_name + ".bin"
        if not os.path.exists(file_path):
            print(f"Error, path {file_path} does not exists")
            return None, None
        scan = np.fromfile(file_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        intensities = scan[:, 3]
        pts = scan[:, 0:3]

        return pts, intensities


    def get_frame_ground_mask(self, number):
        """
        Filters out non ground objects from PCL

            Parameters:
                number (int): number of PCL sequence to be loaded

            Returns:
                mask (numpy.array): array of booleans indicating which points
                    in synchronized PCL are ground

        """
        #labels = self.get_labels(number)
        #mask = np.array(labels == 40) | np.array(labels == 72) | np.array(labels == 44) \
        #    | np.array(labels == 48) | np.array(labels == 49)

        file_name = "0" * int(6 - (len(str(number)))) + str(number)
        file_path = self.path + 'ground_label/' + file_name + ".npy"
        if not os.path.exists(file_path):
            print(f"Error, path {file_path} does not exists")
            return None
        
        mask = np.load(file_path)

        return mask


    def get_frame_without_ground_mask(self, number):
        """
        Filters out ground objects form PCL

            Parameters:
                number (int): number of PCL sequence to be loaded

            Returns:
                mask (numpy.array): array of booleans indicating which points
                    in synchronized PCL are not ground
        """
        #labels = self.get_labels(number)
        #mask = np.array(labels != 40) & np.array(labels != 72) & np.array(labels != 44) \
        #    & np.array(labels != 48) & np.array(labels != 49)

        mask = self.get_frame_ground_mask(number)
        if mask is None:
            return None
        mask = ~mask

        return mask


    def get_frame_without_ground(self, number):
        """
        Loads and synchronizes PCL

            Parameters:
                number (int): number of PCL sequence to be loaded

            Returns:
                pts (numpy.array): N by 3 array of synchronized PCL
                intensities (numpy.array): N by 1 array of intensities of points in PCL
        """
        pts, intensities = self.get_frame(number)
        mask = self.get_frame_without_ground_mask(number)
        if pts is None or mask is None:
            return None, None
        pts = pts[mask]
        intensities = intensities[mask]

        return pts, intensities

    """
    def get_frame_and_remove_ground(self, number):
        pts, _ = self.get_frame_unsynchronized(number)
        pts = remove_ground(pts)
        #inliers, outliers = ransac(pts, origin=[0,0,0])
        inliers, outliers = find_PCA_inliers_outliers(pts)
        pts = pts[outliers]
        pose = self.poses[number]
        pose = np.vstack((pose, [0, 0, 0, 1]))
        pts = self.transform_mat(pts, pose).reshape(-1, 3)

        return pts, []
        """

    def get_moving_cars_mask(self, number):
        labels = self.get_labels(number)
        mask = np.array(labels == 252)
        return mask


    def get_moving_pedestrians_mask(self, number):
        labels = self.get_labels(number)
        mask = np.array(labels == 254)
        return mask


    def get_synchronized_frames(self, first_frame, num_of_frames):
        pts, intens = self.get_frame(first_frame)
        pts = np.hstack((pts, np.ones((pts.shape[0], 1)) * first_frame)) # time first_frame
        for i in range(first_frame + 1, first_frame + num_of_frames):
            new_pts, new_intens = self.get_frame(i)
            new_pts = np.hstack((new_pts, np.ones((new_pts.shape[0], 1)) * i)) # time i
            pts = np.concatenate((pts, new_pts))
            intens = np.concatenate((intens, new_intens))
        return pts, intens


    def get_synchronized_frames_without_ground(self, first_frame, num_of_frames):
        pts, intens = self.get_frame_without_ground(first_frame)
        pts = np.hstack((pts, np.ones((pts.shape[0], 1)) * first_frame)) # time first_frame
        for i in range(first_frame + 1, first_frame + num_of_frames):
            new_pts, new_intens = self.get_frame_without_ground(i)
            new_pts = np.hstack((new_pts, np.ones((new_pts.shape[0], 1)) * i)) # time i
            pts = np.concatenate((pts, new_pts))
            intens = np.concatenate((intens, new_intens))
        return pts, intens


# https://medium.com/@ajithraj_gangadharan/3d-ransac-algorithm-for-lidar-pcd-segmentation-315d2a51351
def ransac(points, origin, close_points_radius=6, max_iterations=10, distance_ratio_threshold=0.2):
    """
    Apply RANSAC plane fitting for ground removal

    Parameters:
        points (numpy.array): 3D points of PCL
        origin (list): coords of ego(lidar)
        close_points_radius (float): radius of sphere around origin where to fit the plane
        max_iterations (int): number of iterations, during each iteration a new
            plane is created
        distance_ratio_threshold (float): distance which decides inliers/outliers
        
    """
    inliers_result = []
    outliers_result = []

    low_points = points[:,2] < -0.5
    points = points[low_points]

    tree = KDTree(points[:,:3])
    close_points = points[tree.query_ball_point(origin, close_points_radius)]
    best_plane = []

    for _ in range(max_iterations):
        # Add 3 random indexes
        random.seed()
        inliers = []
        outliers = []
        n, _ = close_points.shape
        while len(inliers) < 3:
            random_index = random.randint(0, n - 1)
            inliers.append(random_index)

        x1, y1, z1 = close_points[inliers[0]]
        x2, y2, z2 = close_points[inliers[1]]
        x3, y3, z3 = close_points[inliers[2]]
        # Plane Equation --> ax + by + cz + d = 0
        # Value of Constants for inlier plane
        a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d = -(a * x1 + b * y1 + c * z1)
        plane_lenght = max(0.1, np.sqrt(a * a + b * b + c * c))

        for idx, point in enumerate(close_points):
            if idx in inliers:
                # point already used
                continue
            x, y, z = point

            # Calculate the distance of the point to the inlier plane
            distance = np.fabs(a * x + b * y + c * z + d) / plane_lenght
            # Add the point as inlier, if within the threshold distancec ratio
            if distance <= distance_ratio_threshold:
                inliers.append(idx)
            else:
                outliers.append(idx)

        # Update the set for retaining the maximum number of inlier points
        if len(inliers) > len(inliers_result):
            inliers_result = inliers
            outliers_result = outliers
            best_plane = [a,b,c,d,plane_lenght]

        
    # Segregate inliers and outliers from the point cloud
    # inlier_points = points[inliers_result]
    # outlier_points = points[outliers_result]

    # use the best plane for all points (not just close ones)
    a,b,c,d,plane_lenght = best_plane
    inliers_result = []
    outliers_result = []
    for idx, point in enumerate(points):
        x, y, z = point

        # Calculate the distance of the point to the inlier plane
        distance = np.fabs(a * x + b * y + c * z + d) / plane_lenght
        # Add the point as inlier, if within the threshold distancec ratio
        if distance <= distance_ratio_threshold:
            inliers_result.append(idx)
        else:
            outliers_result.append(idx)

    return inliers_result, outliers_result


# https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
def PCA(data, correlation=False, sort=True):
    """ Applies Principal Component Analysis to the data

Parameters
----------
data: array
    The array containing the data. The array must have NxM dimensions, where each
    of the N rows represents a different individual record and each of the M columns
    represents a different variable recorded for that individual record.
        array([
        [V11, ... , V1m],
        ...,
        [Vn1, ... , Vnm]])

correlation(Optional) : bool
        Set the type of matrix to be computed (see Notes):
            If True compute the correlation matrix.
            If False(Default) compute the covariance matrix.
            
sort(Optional) : bool
        Set the order that the eigenvalues/vectors will have
            If True(Default) they will be sorted (from higher value to less).
            If False they won't.
Returns
-------
eigenvalues: (1,M) array
    The eigenvalues of the corresponding matrix.
    
eigenvector: (M,M) array
    The eigenvectors of the corresponding matrix.

Notes
-----
The correlation matrix is a better choice when there are different magnitudes
representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points

Parameters
----------
points: array
    The x,y,z coordinates corresponding to the points from which we want
    to define the best fitting plane. Expected format:
        array([
        [x1,y1,z1],
        ...,
        [xn,yn,zn]])
        
equation(Optional) : bool
        Set the oputput plane format:
            If True return the a,b,c,d coefficients of the plane.
            If False(Default) return 1 Point and 1 Normal vector.
Returns
-------
a, b, c, d : float
    The coefficients solving the plane equation.

or

point, normal: array
    The plane defined by 1 Point and 1 Normal vector. With format:
    array([Px,Py,Pz]), array([Nx,Ny,Nz])
    
    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:, 2]

    #: get a point from the plane
    point = np.mean(points, axis=0)

    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal


def find_PCA_inliers_outliers(points, distance_ratio_threshold=0.2, equation=False):
    """
    Execute PCA ground removal

    Parameters:
        points (numpy.array): N by 3 array of 3D points
        distance_ratio_threshold (float): threshold which decides inliers/outliers
        equation(bool) : Set the oputput plane format:
            If True return the a,b,c,d coefficients of the plane.
            If False(Default) return 1 Point and 1 Normal vector.

    Returns:
        inliers_idx, outliers_idx (list): indicies of inliers/outliers
    """
    low_points = points[:,2] < -0.5 
    points = points[low_points]
    tree = KDTree(points)
    close_points = tree.query_ball_point([0,0,0], 7)
   
    a, b, c, d = best_fitting_plane(points[close_points], True)
    #a, b, c, d = best_fitting_plane(points, True)

    inliers_idx = []
    outliers_idx = []

    plane_lenght = max(0.1, np.sqrt(a * a + b * b + c * c))

    for idx, point in enumerate(points):
        x, y, z = point
        distance = np.fabs(a * x + b * y + c * z + d) / plane_lenght
        if distance <= distance_ratio_threshold:
            inliers_idx.append(idx)
        else:
            outliers_idx.append(idx)

    return inliers_idx, outliers_idx


def calculate_metrics(predicted, truth):
    """
    Computes precision, recall, iou from two arrays

    Parameters:
        predicted (numpy.array): array of bools or ints, 1 means ground,
            0 means not ground

        truth (numpy.array): array of bools or ints, 1 means ground,
            0 means not ground

    Returns:
        precision (float)
        recall (float)
        iou (float)
    """
    predicted = predicted.copy().astype(int)
    truth = truth.copy().astype(int)
    TP = np.logical_and(predicted == 1, truth == 1).sum()
    # TN = np.logical_and(predicted == 0, truth == 0).sum()
    FP = np.logical_and(predicted == 1, truth == 0).sum()
    FN = np.logical_and(predicted == 0, truth == 1).sum()

    if TP == 0:
        if FP == 0:
            precision = 1
        else:
            precision = 0
        if FN == 0 and FP == 0:
            recall = 1
            iou = 1
        elif FN == 0 and FP != 0:
            recall = 1
            iou = TP / (TP + FP + FN)
        elif FN != 0:
            recall = TP / (TP + FN)
            iou = TP / (TP + FP + FN)
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        iou = TP / (TP + FP + FN)

    return precision, recall, iou
