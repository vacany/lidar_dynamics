import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.path as mpltPath
import matplotlib.pyplot as plt

from exps.utils import timeit

from data.point_clouds.box import boxes_to_corners_3d

class Trajectory():
    def __init__(self, odometry, all_poses=None):
        # odometry array, list Nx [x,y,z, yaw]
        self.odometry = odometry
        self.all_poses = all_poses

    @classmethod
    def from_poses(self, poses):
        '''

        :param poses: array (N, 4,4)
        :return: Trajectory class
        '''

        poses = np.array(poses)

        z_angle = [Rotation.from_matrix(i[:3,:3]).as_euler('xyz')[-1] for i in poses]
        z_angle = np.array(z_angle)

        odometry = poses[:,:3,-1]

        odometry_angle = np.concatenate((odometry, z_angle[:, None]), axis=1)

        # odometry_angle = self.refine_trajectory(odometry_angle)

        return Trajectory(odometry_angle, all_poses=poses)

    @classmethod
    def from_odometry(self, odometry):

        for i in range(1, len(odometry)):

            diff = odometry[i-1][:2] - odometry[i][:2]
            angle = np.arctan2(diff[1], diff[0])  # x,y, BEWARE OF 0!
            odometry[i][3] = angle

            # retrospective
            if i == 1:
                odometry[0][3] = angle

        return Trajectory(odometry=odometry)

    # def _get_bounding_box(self):
    #     # might add other classes etc.
    #     self.ego_bbox = get_ego_bbox()

    def synchronize_poses(self, pose_list, reference_frame):
        ref_pose = pose_list[reference_frame]
        shifted_poses = [np.linalg.inv(ref_pose) @ pose for pose in pose_list]
        return shifted_poses

    @classmethod
    def refine_trajectory(self, odometry_angle):
        for i in range(1, len(odometry_angle)):
            if abs(odometry_angle[i][:2].sum() - odometry_angle[i-1][:2].sum()) < 0.02:
                # Angle & position
                odometry_angle[i][:2] = odometry_angle[i-1][:2]
                odometry_angle[i][3] = odometry_angle[i-1][3]

        return odometry_angle

    @classmethod
    def create_path_from_odo(self, odometry, bbox=None):
        '''
        Works only for smooth driving?
        :param odometry: list of xy coordinates
        :param bbox: x,y,z,l,w,h,yaw of object
        :return: list of path for matplotlib path calculation
        '''
        # if bbox is None:
        #     bbox = get_ego_bbox()

        l, w = bbox[3], bbox[4]
        path_forth = []
        path_back = []
        for i in range(len(odometry)):
            bbox = np.array((odometry[i][0], odometry[i][1], bbox[2], l, w, bbox[5], odometry[i][3]))
            corners = boxes_to_corners_3d(bbox[None, :])
            # only xy coordinates
            corners = corners[0][:4, :2]
            # Add to path
            path_forth.append(corners[[0, 3]])
            path_back.append(corners[[2, 1]])  # fixed in a way, it is created

        path_back.reverse()

        point_path = path_forth + path_back
        point_path = np.concatenate(point_path)

        return point_path

    def plot_trajectory(self):
        fig, ax = plt.subplots()
        # Iterate from pairs(first to second)
        for t in range(1, len(self.odometry)):
            ax.annotate(f"{t}", xy=(self.odometry[t][0], self.odometry[t][1]), xytext=(self.odometry[t - 1][0], self.odometry[t - 1][1]),
                    arrowprops=dict(arrowstyle="->"))

        plt.plot(self.odometry[:,0], self.odometry[:,1], 'r.')
        plt.show()

def _test_path_labeling(odometry=None, time_step=0.4, points=False):
    if odometry is None:
        odometry = np.array([[-1, -1], [0, 0], [5, 5], [10, 10], [20, 20]])

    point_path = Trajectory.create_path_from_odo(odometry)

    polygon = point_path
    # random points set of points to test
    if points:
        N = 10000
        points = np.random.rand(N, 2) * 20 - 10

        mask = points_in_path(points, polygon)
        plt.plot(points[mask][:, 0], points[mask][:, 1], 'rx')
        plt.plot(points[mask == False][:, 0], points[mask == False][:, 1], 'bx')

    polygon = np.array(polygon)


    for i in range(len(polygon)):
        plt.plot(polygon[i, 0], polygon[i, 1], 'yo')
        plt.show(block=False)
        plt.pause(time_step)
    plt.show()

def intermediates(p1, p2, nb_points=8):
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] +  i * y_spacing]
            for i in range(1, nb_points+1)]


def get_3d_traj_from_positions(odometry_positions, nb_points=3):
    xy = odometry_positions[:,:2]
    z = odometry_positions[:,2:3]

    traj_xy = [intermediates(pos[0], pos[1], nb_points=nb_points) for pos in zip(xy[:-1],xy[1:])]
    traj_xy = np.concatenate(traj_xy)

    traj_z = np.linspace(z[0], z[1], num=len(traj_xy))
    traj_z = np.full_like(traj_z, z[0])

    traj = np.concatenate((traj_xy, traj_z), axis=1)
    return traj


def line_traj(odo):
    ''' Create a line of discretized points from odometry '''
    odo = np.array(odo, dtype=np.int)
    traj = np.array([(odo[0])], dtype=np.int)

    for i in range(1, len(odo)):

        num_of_point = np.max((abs(odo[i][0] - odo[i - 1][0]), abs(odo[i][1] - odo[i - 1][1])))

        if num_of_point == 0:
            num_of_point = 1

        points = intermediates(odo[i-1], odo[i], num_of_point)
        points = np.array(points)
        traj = np.concatenate((traj, np.array([odo[i-1]])), axis=0)
        traj = np.concatenate((traj, points), axis=0)   # intermediates only

    if len(odo) != 1:
        traj = np.concatenate((traj, np.array([odo[i]])), axis=0)
    else:
        traj = odo


    traj = np.array(np.round(traj), dtype=int)

    traj = traj[traj[:, 1].argsort()]

    return traj


@timeit
def points_in_path(points, odometry):
    path = mpltPath.Path(odometry)
    inside = path.contains_points(points)

    return inside


if __name__ == '__main__':
    batch = np.load('../unsupervised_motion/tmp_batch.npy', allow_pickle=True).item()

    poses = batch['all_poses']

    Ego_traj = Trajectory.from_poses(poses[:])
    # _test_path_labeling(Ego_traj.odometry, time_step=0.003)





