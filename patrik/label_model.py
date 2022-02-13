import numpy as np
import torch.utils.data

from data.datasets.hd_map import HD_map
from data.trajectory.trajectory import Trajectory
from data.point_clouds.box import connect_3d_corners
from data.point_clouds.box import Bounding_Box_Fitter
from exps.utils import timeit

from cool_model import input_features
from cool_model import clz_features


class Lidar_Detector():
    ''' Class for predictions on one batch of multiple point clouds '''
    def __init__(self, cfg):
        self.cfg = cfg


    def remove_geometrical_features(self, pcl):
        # Geometry
        pcl = input_features.calibrate_height(
                pcl,
                sensor_height=self.cfg['SENSOR_HEIGHT']
        )

        ground_and_dist_mask = input_features.remove_ground_and_distance(
                pcl,
                ground_removal_noise=self.cfg['GROUND_ADD']
        )

        return ground_and_dist_mask

    def cluster_by_intensity(self, pcl):
        # Intensity
        inten_mask = input_features.get_reflective_surface(
            pcl,
            inten_thres=self.cfg['INTEN_THRESH']
        )

        traffic_signs_mask = input_features.get_traffic_sign(
            pcl,
            inten_mask=inten_mask,
            min_height=self.cfg['TRAFFIC_SIGN']['MIN_HEIGHT']
        )

        plate_mask = input_features.get_license_plate(
            pcl,
            inten_mask,
            max_height=self.cfg['LICENSE_PLATE']['MAX_HEIGHT']
        )


        # Clustering
        clusters = input_features.get_vehicle_plates(
                pcl,
                plate_mask,
                eps=self.cfg['LICENSE_PLATE']['EPSILON'],
                min_samples=self.cfg['LICENSE_PLATE']['MIN_SAMPLES']
        )

        plate_clusters = input_features.eliminate_oversized_plates(
                pcl,
                clusters,
                max_plate_z_var=self.cfg['LICENSE_PLATE']['MAX_Z_VAR']
        )

        return plate_clusters, traffic_signs_mask


    def clusters_inten_objects(self, pcl):

        plates_clusters, traffic_signs_mask = self.cluster_by_intensity(pcl)

        # Vehicle operations
        VEHICLE = clz_features.Basic_Clustering(
                eps=self.cfg['VEHICLE']['EPSILON'],
                min_samples=self.cfg['VEHICLE']['MIN_SAMPLES']
        )
# bookmarks
        vehicle_ids = VEHICLE.candidates_by_mask(
                pcl,
                plates_clusters,
                min_size=self.cfg['VEHICLE']['MIN_SIZE'],
                max_size=self.cfg['VEHICLE']['MAX_SIZE']
        )

        # Cyclist operations - Same clustering as Vehicle
        # CYCLIST = clz_features.Basic_Clustering(
        #         eps=self.cfg['CYCLIST']['EPSILON'],
        #         min_samples=self.cfg['CYCLIST']['MIN_SAMPLES']
        # )

        cyclist_ids = VEHICLE.candidates_by_mask(
                pcl,
                plates_clusters,
                min_size=self.cfg['CYCLIST']['MIN_SIZE'],
                max_size=self.cfg['CYCLIST']['MAX_SIZE']
        )

        return vehicle_ids, cyclist_ids

    def cluster_buildings(self, pcl):

        BUILDING = clz_features.Basic_Clustering(
                eps=self.cfg['BUILDING']['EPSILON'],
                min_samples=self.cfg['BUILDING']['MIN_SAMPLES']
        )

        building_ids = BUILDING.candidates_by_mask(
                pcl,
                mask_ids=pcl[:, 2] > self.cfg['BUILDING']['MIN_DISTING_HEIGHT'],
                min_size=self.cfg['BUILDING']['MIN_SIZE']
        )

        return building_ids

    def fit_boxes_on_clusters(self, pcl_global, clusters):
        ''' Whole point cloud for height integration '''

        bbox_list = []

        for num, id in enumerate(np.unique(clusters)):
            if id == -1: continue

            mask = clusters == id

            pcl_cluster = pcl_global[mask]

            fitted_bbox = Bounding_Box_Fitter.fit_box(pcl_cluster, pcl_global)

            bbox_list.append(fitted_bbox)

        boxes = np.stack(bbox_list) if len(bbox_list) else None

        ''' Sign Bodies '''
        # seg_label[clusters['sign_bodies']] = self.config['Pole']['label']
        # seg_label[clusters['traffic_signs']] = self.config['Traffic_Sign']['label']
        # seg_label[clusters['buildings'] > 0] = self.config['Building']['label']

        return boxes

    @timeit
    def fit_pedestrians(self, orig_pcl):

        pts_ = orig_pcl[:,:3]
        times = orig_pcl[:,4:5]
        mask = (pts_[:,2] > self.cfg["PEDESTRIAN"]["PCL_MIN_HEIGHT"]) & (pts_[:,2] < self.cfg["PEDESTRIAN"]["PCL_MAX_HEIGHT"])
        cropped_pts = pts_[mask]
        cropped_times = times[mask]


        predictions = clz_features.find_pedestrians(pts_[:,:3], times, cropped_pts, cropped_times)

        dynamic_mask, clustering, dynamic_clusters, differences, centroids_final = predictions

        # mask clustering
        pedestrian_ids = np.zeros(clustering.labels_.shape, dtype=np.int) - 1

        for id in dynamic_clusters:
            pedestrian_ids[clustering.labels_ == id] = id
        # pedestrian_ids = - np.ones(cropped_pts.shape[0])

        return pedestrian_ids, mask

    @timeit
    def run_clustering(self, batch):
        local_pcl = batch['local_points']

        geo_mask = self.remove_geometrical_features(local_pcl.copy())  # Remove ground and distance

        time_mask = input_features.mask_frames_by_time(
                local_pcl,
                from_time=batch['frame'] - self.cfg['NBR_OF_FRAMES'],
                till_time=batch['frame'] + 1
        )

        accum_mask = geo_mask & time_mask

        pcl = local_pcl[accum_mask].copy()

        ''' all happens on augmented point cloud with ground and distance removed '''
        ''' Clustering '''

        plates, traffic_sign = self.cluster_by_intensity(pcl)

        vehicle_ids, cyclist_ids = self.clusters_inten_objects(pcl)
        pedestrian_ids, ped_valid_mask = self.fit_pedestrians(local_pcl)
        building_ids = self.cluster_buildings(pcl)

        segmentation_clusters = {}

        segmentation_clusters['cluster_valid_indices'] = input_features.get_indices_by_mask(accum_mask)
        segmentation_clusters['ped_valid_indices'] = input_features.get_indices_by_mask(ped_valid_mask)
        segmentation_clusters['cluster_time_mask'] = time_mask
        segmentation_clusters['pcl_mask'] = accum_mask
        segmentation_clusters['geo_mask'] = geo_mask

        for key, values in zip(["BUILDING", "VEHICLE", "PEDESTRIAN", "CYCLIST", "TRAFFIC_SIGN"],
                               [building_ids, vehicle_ids, pedestrian_ids, cyclist_ids, traffic_sign]):

            segmentation_clusters[key] = values

        batch['clusters'] = segmentation_clusters

        return batch

        # TODO
        #  Repair bounding boxes,
        #  Time corresponding bounding boxes,
        #  relabel by box dimensions,
        #  HD_map insertion



class Annotator():
    ''' Class for combining HD_map and data from all time'''
    def __init__(self, cfg):

        self.cfg = cfg
        self.model = Lidar_Detector(cfg)

    def initiate_hd_map(self, all_poses):
        Ego_traj = Trajectory.from_poses(all_poses)
        odo = Ego_traj.odometry
        self.hd_map = HD_map(odo=odo)

    def plot_bounding_boxes(self, batch, path):
        bbox_names = ['vehicle_boxes', 'cyclist_boxes', 'pedestrian_boxes']
        cluster_names = ['vehicle_clusters', 'cyclist_clusters', 'pedestrian_clusters']

        for box_type, cluster_type in zip(bbox_names, cluster_names):

            clusters = batch[cluster_type]
            boxes = batch[box_type]

            if boxes is None:
                continue
            # Use this to relabel
            if box_type.startswith('vehicle'):
                class_ = self.cfg['VEHICLE']['LABEL']

            elif box_type.startswith('cyclist'):
                class_ = self.cfg['CYCLIST']['LABEL']

            elif box_type.startswith('pedestrian'):
                class_ = self.cfg['PEDESTRIAN']['LABEL']

            for box in boxes:
                box = np.insert(box, 7, class_)
                valid_ids = np.unique(clusters)

                for id in valid_ids:
                    if id == -1: continue

                    if box_type.startswith('pedestrian'):
                        pcl_cluster = batch['peds_point_cloud'][clusters==id]
                    else:
                        pcl_cluster = batch['edited_point_cloud'][clusters==id]

                    img_path = path + f'_{id}.png'
                    Bounding_Box_Fitter.plot(pcl_cluster, box, path=img_path)


    # TODO
    def relabel_by_box(self):
        pass

    # TODO
    def label_from_predictions(self):
        pass



    def run_annotation(self, dataset, sequence):
        # TODO below zero frames
        for frame in range(self.cfg["PEDESTRIAN"]["NBR_OF_FRAMES"], len(dataset)):
            print(f"Frame: {frame} -------")
            if self.cfg['DATASET_NAME'] == 'semantic-kitti':
                batch = dataset.get_multi_frame(frame)
            if self.cfg['DATASET_NAME'] == 'once':
                batch = dataset.get_sample(frame)

            batch = self.model.run_clustering(batch)


            self.save_data(batch, f'{self.cfg["exp_root"]}/data/{sequence}/gen_labels/{batch["frame"]:06d}.npz')


        # self.hd_map.store_hd_map(f'{self.cfg["exp_root"]}/data/{sequence:02d}/hd_map.npz')




    def save_data(self, batch, path):
        save_list = ['clusters',
                     ]

        # data = {}

        # for key in save_list:
        #     data[key] = batch[key]

        data = self._get_data_dict_from_results(batch)
        bbox_array = self._get_bboxes_from_data_dict_results(batch, data)

        for time, values in data.items():
            if bbox_array is None:
                data[time]['boxes'] = None
            else:
                data[time]['boxes'] = bbox_array[bbox_array[:, 8] == time]
            data[time]['pose'] = batch['poses'][-1]

        np.savez_compressed(path, data)

    def _get_bboxes_from_data_dict_results(self, batch, data, plot=True):
        """

        :param batch:
        :param data: data_dict from _get_data_dict_from_results function with clusters.

        :return: Bounding box array (N, x,y,z,l,w,h,yaw,cls,time,id) with all times
        """
        global_points = batch['points']
        clusters = data
        bbox_list = []

        for t in np.unique(batch['points'][:,4]):
            if t != batch['frame']: continue

            time_mask = global_points[:, 4] == t

            for class_ in self.cfg['OBJ_DET_CLASSES']:

                for id in np.unique(clusters[int(t)][class_]):
                    if id == -1: continue
                    # iterate id clusters from valid ids

                    cluster_mask = clusters[int(t)][class_] == id
                    point_cloud = global_points[time_mask]
                    pcl_cluster = point_cloud[cluster_mask]

                    if len(pcl_cluster) < 3:    # at least 3 points
                        continue

                    box = Bounding_Box_Fitter.fit_box(pcl_cluster, point_cloud)

                    if box is not None:     # Does not go through class check
                        box = np.insert(box, 7, self.cfg[class_]['LABEL'])
                        box = np.insert(box, 8, t)
                        box = np.insert(box, 9, id)

                        bbox_list.append(box)

                    if plot:
                        print(f"Time: {int(t)} \t ID: {int(id)} \t CLASS: {class_}")
                        path = f"{self.cfg['exp_root']}/images/{int(batch['seq_nbr']):04d}_{int(t):04d}_{int(id)}.png"
                        Bounding_Box_Fitter.plot(pcl_cluster, box, path=path)

        if len(bbox_list) > 0:
            return np.stack(bbox_list)
        else:
            return None

    def _get_data_dict_from_results(self, batch):
        """

        :param batch:
        :return: data_dict with separated clustering ids and labels by time.
        The data_dict is keyed by frame number

        """
        # functions to work with batch
        times = batch['points'][:, 4]
        res = batch['clusters']

        time_list = np.unique(times)

        new_data = {}

        for t in time_list:

            cur_time = times == t
            tmp_data = {}


            cls_list = ['BUILDING', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'TRAFFIC_SIGN']
            for cls in cls_list:
                label_nbr = self.cfg[cls]['LABEL']

                object_ids = res[cls]
                point_cloud = batch['points']
                label = - np.ones(point_cloud.shape[0])


                if cls == 'PEDESTRIAN':
                    label[res['ped_valid_indices']] = object_ids
                elif cls == 'TRAFFIC_SIGN':
                    label[res['cluster_valid_indices']] = np.array(res[cls], dtype=np.int)
                else:
                    label[res['cluster_valid_indices']] = object_ids

                label = label[cur_time]

                seg_label = - np.ones(label.shape[0])
                seg_label[label >= 0] = label_nbr

                tmp_data[cls] = label
                tmp_data['seg_label'] = seg_label

            new_data[int(t)] = tmp_data

        return new_data

    def _strip_dict_item(self, npz_file):
        return npz_file['arr_0'].item()

    def after_all_dataset(self, exp_root, sequence):
        self.hd_map.store_hd_map(f'{exp_root}/{sequence:02d}/hd_map.npz')


if __name__ == '__main__':
    from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
    from exps.rw import load_yaml
    sequence = 18
    prev = 10
    dataset = SemKittiDataset(prev=prev, sequence=[sequence])

    batch = dataset.get_multi_frame(50)

    cfg = load_yaml('semantic-kitti.yaml')

    labeler = Annotator(cfg)
    # labeler.
