import numpy as np
import os
import yaml
import random

PATH = "../../semantic_kitti_data/sequences/18/"
POSES = np.loadtxt(PATH + "poses.txt")
POSES = POSES.reshape(-1, 3, 4)


def read_calibration_file(path):
    file = os.path.join(path,"calib.txt")
    calibration_tr = ""
    with open(file) as f:
        for ln in f:
            if ln.startswith("Tr: "):
                calibration_tr = ln[4:-1]
    
    calibration_tr = np.array(calibration_tr.split(' '), dtype=np.float64)
    calibration_tr = calibration_tr.reshape(3,4)
    calibration_tr = np.vstack((calibration_tr, np.array([0,0,0,1])))
    
    return calibration_tr

def transform_mat(_pts, pose):
    """
    multiply matrix of 3d points by transformation matrix, which will result in original 3d points
    being synchronized across multiple sequences, meaning a rigid object will stay in place even though
    the LIDAR is moving
     """
    # 01
    """calibration_tr = np.array([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
                               -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
                               9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
                               0, 0, 0, 1])"""
    # 05
    calibration_tr = np.array([-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
                               -6.481465826011e-03, 8.051860151134e-03,-9.999466081774e-01, -7.337429464231e-02,
                               9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
                               0, 0, 0, 1])
    # 07
    """calibration_tr = np.array([-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
                               -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
                               9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
                               0, 0, 0, 1])"""
    # tr = calibration_tr.reshape(4, 4)
    tr = read_calibration_file(PATH)
    tr_inv = np.linalg.inv(tr)
    pose = np.matmul(tr_inv, np.matmul(pose, tr))
    n, _ = _pts.shape
    x = np.hstack((_pts, np.ones((n, 1))))
    x = np.matmul(pose, x.transpose()).transpose()
    return x[:, 0:3]


def get_frame(number):
    """
    Loads and synchronizes PCL

        Parameters:
            number (int): number of PCL sequence to be loaded

        Returns:
            pts (numpy.array): N by 3 array of synchronized PCL
            intensities (numpy.array): N by 1 array of intensities of points in PCL
    """
    file_name = "0" * int(6 - (len(str(number)))) + str(number)
    scan = np.fromfile(PATH + "velodyne/" + file_name + ".bin", dtype=np.float32)
    scan = scan.reshape((-1, 4))
    intensities = scan[:, 3]
    pts = scan[:, 0:3]

    pose = POSES[number]
    pose = np.vstack((pose, [0, 0, 0, 1]))
    pts = transform_mat(pts, pose).reshape(-1, 3)

    return pts, intensities


def get_frame_unsynchronized(number):
    """
    Loads PCL without synchronization

        Parameters:
            number (int): number of PCL sequence to be loaded

        Returns:
            pts (numpy.array): N by 3 array of  PCL
            intensities (numpy.array): N by 1 array of intensities of points in PCL
    """
    file_name = "0" * int(6 - (len(str(number)))) + str(number)
    scan = np.fromfile(PATH + "velodyne/" + file_name + ".bin", dtype=np.float32)
    scan = scan.reshape((-1, 4))
    intensities = scan[:, 3]
    pts = scan[:, 0:3]

    return pts, intensities


def get_frame_ground_mask(number):
    """
    Filters out non ground objects from PCL

        Parameters:
            number (int): number of PCL sequence to be loaded

        Returns:
            mask (numpy.array): array of booleans indicating which points
                in synchronized PCL are ground

    """
    # pts, intensities = get_frame(number)
    file_name = "0" * int(6 - (len(str(number)))) + str(number)
    labels = np.fromfile(PATH + "labels/" + file_name + ".label", dtype=np.uint32)

    mask = np.array(labels == 40) | np.array(labels == 72) | np.array(labels == 44) \
           | np.array(labels == 48) | np.array(labels == 49)

    return mask


def get_frame_without_ground_mask(number):
    """
    Filters out ground objects form PCL

        Parameters:
            number (int): number of PCL sequence to be loaded

        Returns:
            mask (numpy.array): array of booleans indicating which points
                in synchronized PCL are not ground
    """
    # pts, intensities = get_frame(number)
    file_name = "0" * int(6 - (len(str(number)))) + str(number)
    labels = np.fromfile(PATH + "labels/" + file_name + ".label", dtype=np.uint32)

    mask = np.array(labels != 40) & np.array(labels != 72) & np.array(labels != 44) \
           & np.array(labels != 48) & np.array(labels != 49)

    return mask


def get_frame_without_ground(number):
    """
    Loads and synchronizes PCL

        Parameters:
            number (int): number of PCL sequence to be loaded

        Returns:
            pts (numpy.array): N by 3 array of synchronized PCL
            intensities (numpy.array): N by 1 array of intensities of points in PCL
    """
    file_name = "0" * int(6 - (len(str(number)))) + str(number)
    scan = np.fromfile(PATH + "velodyne/" + file_name + ".bin", dtype=np.float32)
    scan = scan.reshape((-1, 4))
    intensities = scan[:, 3]
    pts = scan[:, 0:3]

    pose = POSES[number]
    pose = np.vstack((pose, [0, 0, 0, 1]))
    pts = transform_mat(pts, pose).reshape(-1, 3)

    mask = get_frame_without_ground_mask(number)
    pts = pts[mask]
    intensities = intensities[mask]

    return pts, intensities


def get_dynamic_points_mask(number):
    file_name = "0" * int(6 - (len(str(number)))) + str(number)
    labels = np.fromfile(PATH + "labels/" + file_name + ".label", dtype=np.uint32)

    mask = np.array(labels != 40) & np.array(labels != 72) & np.array(labels != 44) \
           & np.array(labels != 48) & np.array(labels != 49) & np.array(labels != 0) \
           & np.array(labels != 51) & np.array(labels != 81) & np.array(labels != 80) \
           & np.array(labels != 70) & np.array(labels != 50) & np.array(labels != 52) \
           & np.array(labels != 252) & np.array(labels != 111)

    return mask


def get_synchronized_frames(first_frame, num_of_frames):
    pts, intens = get_frame(first_frame)
    pts = np.hstack((pts, np.ones((pts.shape[0], 1)) * first_frame)) # time first_frame
    for i in range(first_frame + 1, first_frame + num_of_frames):
        new_pts, new_intens = get_frame(i)
        new_pts = np.hstack((new_pts, np.ones((new_pts.shape[0], 1)) * i)) # time i
        pts = np.concatenate((pts, new_pts))
        intens = np.concatenate((intens, new_intens))
    return pts, intens


def get_synchronized_frames_without_ground(first_frame, num_of_frames):
    pts, intens = get_frame_without_ground(first_frame)
    pts = np.hstack((pts, np.ones((pts.shape[0], 1)) * first_frame)) # time first_frame
    for i in range(first_frame + 1, first_frame + num_of_frames):
        new_pts, new_intens = get_frame_without_ground(i)
        new_pts = np.hstack((new_pts, np.ones((new_pts.shape[0], 1)) * i)) # time i
        pts = np.concatenate((pts, new_pts))
        intens = np.concatenate((intens, new_intens))
    return pts, intens

# https://medium.com/@ajithraj_gangadharan/3d-ransac-algorithm-for-lidar-pcd-segmentation-315d2a51351
def ransac(points, max_iterations, distance_ratio_threshold, min_inliers_to_pass):
    """
    Apply RANSAC plane fitting for ground removal

    Parameters:
        points (numpy.array): 3D points of PCL
        max_iterations (int): number of iterations, during each iteration a new
            plane is created
        distance_ratio_threshold (float): distance which decides inliers/outliers
        min_inliers_to_pass (float):
    """
    inliers_result = []
    outliers_result = []

    for _ in range(max_iterations):
        # Add 3 random indexes
        random.seed()
        inliers = []
        outliers = []
        n, _ = points.shape
        while len(inliers) < 3:
            random_index = random.randint(0, n)
            inliers.append(random_index)

        x1, y1, z1 = points[inliers[0]]
        x2, y2, z2 = points[inliers[1]]
        x3, y3, z3 = points[inliers[2]]
        # Plane Equation --> ax + by + cz + d = 0
        # Value of Constants for inlier plane
        a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d = -(a * x1 + b * y1 + c * z1)
        plane_lenght = max(0.1, np.sqrt(a * a + b * b + c * c))

        for idx, point in enumerate(points):
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

        if len(inliers) >= min_inliers_to_pass:
            break
    # Segregate inliers and outliers from the point cloud
    # inlier_points = points[inliers_result]
    # outlier_points = points[outliers_result]

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
    a, b, c, d = best_fitting_plane(points, True)

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
    TN = np.logical_and(predicted == 0, truth == 0).sum()
    FP = np.logical_and(predicted == 1, truth == 0).sum()
    FN = np.logical_and(predicted == 0, truth == 1).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    iou = TP / (TP + FP + FN)

    return precision, recall, iou
