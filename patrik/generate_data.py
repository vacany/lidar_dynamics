import os
import sys
import numpy as np
from exps.rw import load_yaml
from cool_model.label_model import Annotator
from exps.evaluation.segmentation import IOU
from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
from data.datasets.once.once import Once_dataset

from cool_model import protocol


if __name__ == "__main__":
    # Arguments
    dataset_choice = int(sys.argv[1])
    sequence = int(sys.argv[2])


    # connect
    dataset_list = ['semantic-kitti', 'once', 'nuscenes']
    print(f'Dataset possibilities 0,1,2 --- {dataset_list}')
     #TODO Refactor
    if dataset_choice == 0:
        config = load_yaml('cool_model/semantic-kitti.yaml')
        dataset = SemKittiDataset(prev=config['PEDESTRIAN']['NBR_OF_FRAMES'], sequence=[sequence])
        sequence = f"{sequence:02d}"
        if int(sequence) > 20:
            raise OSError("SEMANTIC KITTI HAS ONLY 20 SEQUNCES!")
    elif dataset_choice == 1:
        config = load_yaml('cool_model/once.yaml')
        sequence = f"{sequence:06d}"
        dataset = Once_dataset(sequences=[sequence])

    else:
        raise NotImplemented('Nuscenes not implemented yet')


    root_dir = protocol.default_root_dir()
    name = 'test'

    exp_root = root_dir + '/run_' + dataset_list[dataset_choice]

    config['exp_root'] = exp_root

    protocol.exp_structure(exp_root=exp_root, sequence=sequence, config=config)

    annotator = Annotator(config)

    # from torch.utils.data import DataLoader
    # annotator = DataLoader(annotator, batch_size=4, num_workers=8)

    # all_poses = dataset.meta_data[sequence]['poses']

    # labeler.initiate_hd_map(all_poses)

    # metric = IOU(num_classes=8, ignore_cls=7)
    # Maybe refactor to remove sequence, but after submission
    annotator.run_annotation(dataset, sequence=sequence)




