import numpy as np

import render
from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
from exps.rw import load_yaml

from data.point_clouds.clustering import driving_scene


dataset = SemKittiDataset(next=0, chosen_seqs=[7])


# config = load_yaml('../clustering/config_values.yaml')
def clustering_annotator(data, config):

    annotator = driving_scene.Annotator(config=config)
    annotator.update(data)
    annotator.frame_clustering()

    return annotator.cluster_mask

config = load_yaml('test_config.yaml')
def function_inten(data, config):
    inten_thres = config['inten_threshold']

    mask = data['points'][:,3] > inten_thres
    mask = np.array(mask, dtype=np.float)
    # data['points'] = data['points'][mask]
    # mask = data['points'][:,3]
    return mask


config = load_yaml('../clustering/config_values.yaml')

from data.point_clouds.grid.bev import mask_out_of_range_coors

def remove_ground(data, config):

    annotator = driving_scene.Annotator(config=config)
    annotator.update(data)
    annotator.config['add_ground_height'] = 1.

    mask = annotator._Annotator__above_ground()
    data['points'] = data['points'][mask]

    # mask = mask_out_of_range_coors(data['points'][:,:3], x_range=(-40,40), y_range=(-40,40))

    data['points'] = data['points'][mask]

    from data.point_clouds.clustering.structure import DBSCAN



    model = DBSCAN(0.3, min_samples=8)
    model.fit(data['points'][:,:3])
    labels = model.labels_

    mask = labels


    # data['points'] = data['points'][mask]
    # mask = mask[mask]
    return mask

from data.point_clouds.ground_removal.ground_segment import ground_segment_velodyne
def segment_ground(data, config):
    config = {"nothing" : 0}
    pcl, label, grid = ground_segment_velodyne(data['local_points'])
    data['points'] = pcl
    mask = label

    return mask

def labels(data, config={'shit' : 0}):
    return data['seg_label']


offset=460

render.Vis_function(dataset=dataset, function=remove_ground, config=config, offset=offset)
