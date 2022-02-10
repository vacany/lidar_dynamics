import numpy as np

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

        pcl = pcl[ground_and_dist_mask]

        return pcl

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



    #TODO
    def verify_by_box(self):
        pass




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
        mask = (pts_[:,2] > -1) & (pts_[:,2] < 1)
        cropped_pts = pts_[mask]
        cropped_times = times[mask]


        predictions = clz_features.find_pedestrians(pts_[:,:3], times, cropped_pts, cropped_times)

        dynamic_mask, clustering, dynamic_clusters, differences, centroids_final = predictions

        # mask clustering
        pedestrian_ids = np.zeros(clustering.labels_.shape, dtype=np.int) - 1

        for id in dynamic_clusters:
            pedestrian_ids[clustering.labels_ == id] = id

        return pedestrian_ids, mask


    def run_inference(self, batch):
        cluster_times = list(range(batch['frame'] - self.cfg['NBR_OF_FRAMES'], batch['frame'] + 1))
        ped_times = list(range(batch['frame'] - self.cfg['PEDESTRIAN']['NBR_OF_FRAMES'], batch['frame']))

        global_pcl = batch['points']
        local_pcl = batch['local_points']

        pcl = self.remove_geometrical_features(local_pcl.copy())  # Remove ground and distance
        ped_pcl = self.remove_geometrical_features(local_pcl.copy())  # Remove ground and distance


        pcl_mask = input_features.mask_frames_by_time(
                pcl,
                from_time=cluster_times[0],
                till_time=cluster_times[-1]
        )

        ped_pcl_mask = input_features.mask_frames_by_time(
                pcl,
                from_time=ped_times[0],
                till_time=ped_times[-1]
        )

        ped_pcl = ped_pcl[ped_pcl_mask]
        pcl = pcl[pcl_mask]

        ''' all hapens on augmented point cloud with ground and distance removed '''
        ''' Clustering '''

        plates, traffic_sign = self.cluster_by_intensity(pcl)

        vehicle_ids, cyclist_ids = self.clusters_inten_objects(pcl)
        pedestrian_ids, ped_mask = self.fit_pedestrians(batch['local_points'])
        building_ids = self.cluster_buildings(pcl)

        #TODO Now I am fitting box to all times
        vehicle_boxes = self.fit_boxes_on_clusters(pcl, clusters=vehicle_ids)
        cyclist_boxes = self.fit_boxes_on_clusters(pcl, clusters=cyclist_ids)
        pedestrian_boxes = self.fit_boxes_on_clusters(batch['points'][ped_mask], clusters=pedestrian_ids)

        ''' Saving to batch '''
        batch['edited_point_cloud'] = pcl
        batch['peds_point_cloud'] = batch['points'][ped_mask]

        batch['vehicle_boxes'] = vehicle_boxes
        batch['vehicle_clusters'] = vehicle_ids

        batch['cyclist_boxes'] = cyclist_boxes
        batch['cyclist_clusters'] = cyclist_ids

        batch['pedestrian_boxes'] = pedestrian_boxes
        batch['pedestrian_clusters'] = pedestrian_ids

        batch['building_clusters'] = building_ids
        batch['traffic_sign'] = traffic_sign


        return batch

        # TODO
        #  Repair bounding boxes,
        #  Time corresponding bounding boxes,
        #  relabel by box dimensions,
        #  HD_map insertion



class Annotator(Lidar_Detector):
    ''' Class for combining HD_map and data from all time'''
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

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
        for frame in range(20, len(dataset)):
            print(f"Frame: {frame} -------")
            if self.cfg['DATASET_NAME'] == 'semantic-kitti':
                batch = dataset.get_multi_frame(frame)
            if self.cfg['DATASET_NAME'] == 'once':
                batch = dataset.get_sample(frame)

            batch = self.run_inference(batch)


            self.hd_map.all_bounding_boxes.append(batch['vehicle_boxes'])
            self.hd_map.all_bounding_boxes.append(batch['cyclist_boxes'])
            self.hd_map.all_bounding_boxes.append(batch['pedestrian_boxes'])

            self.plot_bounding_boxes(batch, path=f"{self.cfg['exp_root']}/images/{batch['frame']:06d}")
            self.save_data(batch, f'{self.cfg["exp_root"]}/data/{sequence:02d}/gen_labels/{batch["frame"]:06d}.npz')


        self.hd_map.store_hd_map(f'{self.cfg["exp_root"]}/data/{sequence:02d}/hd_map.npz')


    def save_data(self, batch, path):
        save_list = ['vehicle_boxes',
                     'vehicle_clusters',
                     'cyclist_boxes',
                     'cyclist_clusters',
                     'pedestrian_boxes',
                     'pedestrian_clusters',
                     'building_clusters',
                     'traffic_sign'
                     ]

        data = {}

        for key in save_list:
            data[key] = batch[key]

        np.savez(path, **data)

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

    labeler = Lidar_Detector(cfg)
    labeler.run_inference(batch)
