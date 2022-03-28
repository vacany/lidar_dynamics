import os
import numpy as np
import raycasting
import yaml
from ground_removal import Ground_removal

DATA_PATH = '../../../semantic_kitti_data/sequences/'

if __name__ == '__main__':
    sequences = os.listdir(DATA_PATH)
    sequences.sort()
    print(f"found {sequences}")
    for sequence in sequences:
        if not sequence.isnumeric():
            continue
        curr_path = DATA_PATH + sequence + '/'
        velodyne = os.path.join(curr_path, 'velodyne')
        ground_label = os.path.join(curr_path, 'ground_label')
        predictions = os.path.join(curr_path, 'predictions_raycast')
        if not os.path.isdir(predictions):
            os.mkdir(predictions)
        scans = os.listdir(velodyne)
        scans.sort()
        num_of_scans = len(scans)
        num_of_ground_labels = len(os.listdir(ground_label))
        if num_of_ground_labels < num_of_scans:
            print(f"Error, sequence {sequence} has {num_of_ground_labels} of ground labels and {num_of_scans} of scans")
            continue

        with open('../raycast_config.yaml', 'r') as f:
            config = yaml.safe_load(f)    

        dataloader = Ground_removal(sequence, curr_path)
        rp_car = raycasting.RaycastPredictor(config['CAR'], dataloader)
        rp_ped = raycasting.RaycastPredictor(config['PEDESTRIAN'], dataloader)

        for scan in scans:
            number = int(scan.split('.')[0])
            print(f'sequence: {sequence}, frame: {number}')
            mask = rp_car.predict(number)
            mask = mask | rp_ped.predict(number)
            prediction = np.ones_like(mask, dtype=np.uint32) * 9
            prediction[mask] = 251
            output_name = scan.split('.')[0] + '.label'
            prediction.tofile(os.path.join(predictions, output_name))
            

        