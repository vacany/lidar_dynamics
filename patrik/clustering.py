import numpy as np

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import data.point_clouds.box as box


from cool_model import clz_features

from exps.utils import timeit

class Point_Cloud_Clusterer():
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.show = False

        self.veh_clustering = DBSCAN(config['Vehicle']['min_dist'], min_samples=config['Vehicle']['min_samples'])
        self.sign_clustering = DBSCAN(config['Traffic_Sign']['min_dist'], min_samples=config['Traffic_Sign']['min_samples'])
        self.building_clustering = DBSCAN(config['Building']['min_dist'], min_samples=config['Building']['min_samples'])

        self.box_list = []

    def get_inten_objects(self):

        self.veh_clustering.fit(self.local_pcl[:,:3])
        self.all_clusters = self.veh_clustering.labels_

        # self.vehicle_by_plate_mask = np.zeros(self.all_clusters.shape)
        # self.cyclist_by_plate_mask = np.zeros(self.all_clusters.shape)


    def get_clusters(self):

        self.get_inten_objects()
        self.get_signs_bodies()
        self.get_big_buildings()

        self.clusters = {'all_clusters' : self.all_clusters,
                         'sign_clusters' : self.sign_clusters,
                         'sign_bodies' : self.sign_poles,
                         'traffic_signs' : self.traffic_sighs,
                         'buildings' : self.buildings
                         }

        valid_clusters = self.plate_clusters > -1
        valid_ids = np.unique(self.all_clusters[valid_clusters])

        inten_objects = np.zeros(self.all_clusters.shape) - 1

        for id in valid_ids:
            inten_objects[self.all_clusters == id] = id

        self.clusters['inten_objects'] = inten_objects

        return self.clusters

    def get_signs_bodies(self):
        self.sign_clustering.fit(self.local_pcl[:,:3])
        self.sign_clusters = self.sign_clustering.labels_

        valid_ids = np.unique(self.sign_clusters[self.traffic_sighs])

        self.sign_bodies = np.zeros(self.sign_clusters.shape)

        for id in valid_ids:

            pcl = self.local_pcl[self.sign_clusters == id, :2]

            if clz_features.check_sign_body(pcl,
                                         max_size=self.config['Traffic_Sign']['max_size']):

                self.sign_bodies[self.sign_clusters == id] = id

        self.traffic_sign = (self.sign_bodies > 0) & self.traffic_sighs
        self.sign_poles = (self.sign_bodies > 0) & (self.inten_mask == False)

    def get_big_buildings(self):
        self.buildings = np.zeros(self.all_clusters.shape)
        self.building_clustering.fit(self.local_pcl[:,:3])

        self.building_clusters = self.building_clustering.labels_

        for id in np.unique(self.building_clusters):
            if id == -1: continue

            pcl = self.local_pcl[self.building_clusters == id, :2]


            if clz_features.check_building(pcl,
                                        min_size=self.config['Building']['min_size']):

                self.buildings[self.building_clusters==id] = 1

