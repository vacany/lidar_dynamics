import numpy as np
import torch
import matplotlib.pyplot as plt
import numba
from numba import jit, njit

from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull, Delaunay

from cool_model.minimum_area_box import MinimumBoundingBox
from exps.utils import timeit


box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def connect_3d_corners(bboxes, fill_points=10, add_label=None):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    corners = boxes_to_corners_3d(bboxes)

    point_list = []
    line = np.linspace(0, 1, fill_points)

    for box in corners:
        for i in range(len(box)):
            if i != 3 and i != 7:
                points = box[i] + (box[i + 1] - box[i]) * line[:, None]
                point_list.append(points)
            if i < 4:
                points = box[i] + (box[i + 4] - box[i]) * line[:, None]
                point_list.append(points)
            if i in [3, 7]:
                points = box[i] + (box[i - 3] - box[i]) * line[:, None]
                point_list.append(points)

    points = np.concatenate(point_list)
    if add_label is not None:
        points = np.insert(points, 3, add_label, axis=1)

    return points

def concatenate_box_pcl(boxes, pcl, label, box_label=1):
    box_points = connect_3d_corners(boxes)
    box_points = np.insert(box_points, 3, box_label, axis=1)
    pcl = np.concatenate((pcl[:,:3], box_points[:,:3]))
    label = np.concatenate((label, box_points[:,3]))

    return pcl, label

def get_bbox_points(bboxes):
    '''

    :param bbox: (N,x,y,z,l,w,h,yaw)
    :return: point cloud of box: x,y,z,l
    '''
    bbox_vis = connect_3d_corners(bboxes, fill_points=30)

    bbox_vis = np.insert(bbox_vis, 3, 1, axis=1)  # class label

    return bbox_vis

def get_point_mask(pcl, bbox, x_add=(0., 0.), y_add=(0., 0.), z_add=(0., 0.)):
    '''
    :param pcl: x,y,z ...
    :param bbox: x,y,z,l,w,h,yaw
    :param x_add:
    :param y_add:
    :param z_add:
    :return: Segmentation mask
    '''

    angle = bbox[6]
    Rot_z = np.array(([np.cos(angle), - np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]))
    s = pcl.copy()
    s[:, :3] -= bbox[:3]
    s[:, :3] = (s[:, :3] @ Rot_z)[:, :3]
    size = np.array((-bbox[3]/2, bbox[3]/2, -bbox[4]/2, bbox[4]/2, -bbox[5]/2, bbox[5]/2))
    point_mask = (size[0] - x_add[0] <= s[:, 0]) & (s[:, 0] <= size[1] + x_add[1]) & (size[2] - y_add[0] <= s[:, 1])\
                 & (s[:, 1] <= size[3] + y_add[1]) & (size[4] - z_add[0] <= s[:,2]) & (s[:,2] <= size[5] + z_add[1])

    return point_mask


def extend_height_box(full_pcl, pcl_cluster, box):
    '''

    :param pcl: Full pcl
    :return:
    '''
    z_max = pcl_cluster[:,2].max()
    h_best = 0
    z_best = 0
    max_points = 0

    for h in range(1,30, 1):
        h = h/10
        z = z_max - h / 2
        box[2] = z
        box[5] = h
        mask = get_point_mask(full_pcl, box)
        contained_points = np.sum(mask)

        if max_points < contained_points:
            max_points = contained_points

            z_best = z
            h_best = h

    # if box[7] == self.config['Vehicle']['label']:
        # lenght
        # if box[3] < 3.5:
        #     box[3] = 3.5    # TODO Taken from ego dimensions, do it as param and from inten !!!

    box[2] = z_best + 0.05
    box[5] = h_best
    return box

def reorder_corners(corners):
    # calculate distnace
    dist = ((corners - corners[0]) ** 2).sum(1)
    indices = np.argsort(dist)

    # init, closest, farthrest, and last one
    order = [indices[0], indices[1], indices[3], indices[2], indices[0]]

    corners = corners[order]
    return corners

def centroid_poly(poly):
    T = Delaunay(poly).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = 0

    for m in range(n):
        sp = poly[T[m, :], :]
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)

    return C / np.sum(W)


def show_box_info(pcl, bounding_box, threshold):
    corners = np.array(list((bounding_box.corner_points)))

    corners = reorder_corners(corners)
    print('----------')
    print(f"Nbr of points {len(pcl)}, Approx. distance {np.sqrt(pcl[:, :3] ** 2).mean():.2f}")
    print(f"Paralelel {bounding_box.length_parallel:.2f}, orthogonal {bounding_box.length_orthogonal:.2f}")




def contain_all_points(pcl, corners):
    hull = ConvexHull(pcl[:, :2])

    hullpoints = pcl[hull.vertices, :]

    contain_points = []
    for point in hullpoints:
        a = Point(point[0], point[1])
        b = Polygon([corners[0], corners[1], corners[2], corners[3], corners[0]])
        contain_points.append(b.contains(a))

    all_points = np.all(contain_points)

    return all_points


def calculate_distance_to_box(pcl, corners):
    hull = ConvexHull(pcl[:,:2])

    hullpoints = pcl[hull.vertices, :]

    criterion_list = []

    for point in hullpoints:
        a = Point(point[0], point[1])
        b = Polygon([corners[0], corners[1], corners[2], corners[3], corners[0]])
        dist = a.distance(b.exterior)
        if point[3] > 0.95:
            dist = dist * 1.2
        criterion_list.append(dist)

    criterion = np.max(criterion_list)
    return criterion

def min_area_to_detection_box(box):
    '''

    :param full_pcl: All points for extension
    :param pcl_cluster: Cluster pcl
    :param box: Min Area bounding box with orthogonal and so on
    :param clz: Add class label
    :return:
    '''

    x, y = box.rectangle_center
    h, z = 1.8, 1.0
    l, w = box.length_parallel, box.length_orthogonal
    yaw = box.unit_vector_angle

    bbox = np.array((x, y, z, l, w, h, yaw))

    return bbox


@jit(nopython=True)
def pointinpolygon(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in numba.prange(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i, 0], points[i, 1], polygon)
    return D


class Bounding_Box_Fitter():

    @classmethod
    @timeit
    def fit_box(self, pcl_cluster, pcl_global):

        bounding_box = MinimumBoundingBox(pcl_cluster[:,:2])
        centroid_hull = centroid_poly(pcl_cluster[:, :2])
        best_bbox = min_area_to_detection_box(bounding_box)
        max_dist = 100
        it = 0

        bbox = best_bbox.copy()
        orig_bbox = bbox.copy()


        yaw_angle = 180
        yaw_step = 1
        for x in range(-1, 2):
            bbox[0] = centroid_hull[0] + x / 5

            for y in range(-1, 2):
                bbox[1] = centroid_hull[1] + y / 5

                for yaw in range(0, yaw_angle, yaw_step):
                    bbox[6] = (yaw / 180) * np.pi

                    for l in range(-1, 3):
                        bbox[3] = orig_bbox[3] + l / 5

                        for w in range(-1, 3):
                            bbox[4] = orig_bbox[4] + w / 10

                            it += 1

                            corners = boxes_to_corners_3d(bbox[None, :])
                            corners = corners[0, :5, :2]

                            all_points_inside = parallelpointinpolygon(pcl_cluster, corners)

                            if np.sum(all_points_inside) <= len(all_points_inside) - 3:
                                continue



                            dist = calculate_distance_to_box(pcl_cluster, corners)

                            if max_dist > dist:
                                # print(f"Iter: {it} \t Max Point Distance: {dist:.3f}")
                                max_dist = dist
                                best_bbox = bbox.copy()

        # print(f"MAX_ITER : {it}")
        best_bbox = extend_height_box(pcl_global, pcl_cluster, best_bbox)

        return best_bbox

    @classmethod
    def plot(self, pcl, bbox, path, show=False):

        plt.clf()
        plt.scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 3] > 0.95, cmap='jet', alpha=0.4)

        corners = boxes_to_corners_3d(bbox[None, :])
        corners = corners[0, :5, :2]

        plt.plot(bbox[0], bbox[1], 'y*')
        plt.plot(corners[:, 0], corners[:, 1], 'r-')
        plt.axis('equal')

        plt.title(  f"T: {int(bbox[8])} ; "
                    f"Cls: {int(bbox[7])} ; "
                    f"id: {int(bbox[9])} ;"
                    f"L: {bbox[3]:.2f}m ; "
                    f"W: {bbox[4]:.2f}m ; "
                    f"Yaw: {bbox[5]:.2f} rad")

        if show:
            plt.show()
        else:

            plt.savefig(f'{path}')
        plt.close()




     # from scipy.spatial.transform import Rotation
        # from sklearn.linear_model import RANSACRegressor
        # from sklearn.cluster import DBSCAN

        # cluster_fit = DBSCAN(eps=0.5, min_samples=2)
        # min_x = 100
        # for i in range(0, yaw_step):
        #     i = i / yaw_step * 2 * np.pi
        #     rot_mat = Rotation.from_rotvec(rotvec=np.array((0,0,i)))
        #     rot_mat = rot_mat.as_matrix()
        #
        #     tmp_pcl = pcl_cluster[:,:2] - (centroid_hull[0], centroid_hull[1])
        #     tmp_pcl = np.insert(tmp_pcl, 2, 1, axis=1)
        #
        #     tmp_pcl = tmp_pcl @ rot_mat
        #
        #     inten_pcl = tmp_pcl[pcl_cluster[:,3] > 0.95]
        #     tmp_pcl = inten_pcl
        #     x_var = np.abs(tmp_pcl[:,0].max() - tmp_pcl[:,0].min())
        #
        #
        #     if min_x > x_var:
        #         min_x = x_var
        #         plt.clf()

                # plt.scatter(tmp_pcl[:,0], tmp_pcl[:,1], c=pcl_cluster[:,3] > 0.95, cmap='jet')

                # ransac = RANSACRegressor(min_samples=len(tmp_pcl), residual_threshold = 0.5)

                # b, k = np.polynomial.polynomial.polyfit(tmp_pcl[:,0], tmp_pcl[:,1], 1)
                # ransac.fit(tmp_pcl[:,0][:,None], tmp_pcl[:,1])
                # inlier = ransac.inlier_mask_
                # outlier = np.logical_not(inlier)

                # y_ransac = ransac.predict(tmp_pcl[:,0][:, np.newaxis])
        #
        #         cluster_fit.fit(tmp_pcl[:,:2])
        #
        #         inten_centroid = box.centroid_poly(tmp_pcl[:, :2])
        #
        #         best_pcl = tmp_pcl
        #         best_rot_mat = rot_mat
        #
        # tmp_pcl = best_pcl
        # centroid = np.array((0,0))
        # diff = centroid - tmp_pcl[:, :2]
        #
        # symmetry_pcl = tmp_pcl[:, :2] + 2 * diff
        #
        # tmp_pcl = np.concatenate((tmp_pcl[:,:2], symmetry_pcl))
        #
        # plt.scatter(tmp_pcl[:, 0], tmp_pcl[:, 1], cmap='jet')
        # plt.plot(inten_centroid[0], inten_centroid[1], 'g*')
        # plt.plot(centroid[0], centroid[1], 'y*')
        #
        # bbox = box.MinimumBoundingBox(tmp_pcl[:,:2])
        # bbox = box.min_area_to_detection_box(bbox)
        #
        # best_bbox = bbox
        #
        # best_bbox[:2] += (centroid_hull[0], centroid_hull[1])
        #
        # inv_rot = Rotation.from_matrix(best_rot_mat).as_rotvec()
        # best_bbox[6] -= inv_rot[2]
        #
        # plt.axis('equal')
        # plt.show(block=False)
        # plt.pause(0.002)
        #
        #
        #
        # plt.plot(bbox[0], bbox[1], 'y*')
        # plt.show()
