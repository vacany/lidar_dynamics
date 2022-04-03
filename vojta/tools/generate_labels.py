import os
import numpy as np
import raycasting
import yaml
from ground_removal import Ground_removal
import sys

DATA_PATH = '../../../semantic_kitti_data/sequences/'

def generate_labels_for_sequence(sequence):
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
        return

    with open('../raycast_config.yaml', 'r') as f:
        config = yaml.safe_load(f)    

    dataloader = Ground_removal(sequence, curr_path)
    rp_car = raycasting.RaycastPredictor(config['CAR'], dataloader)
    rp_ped = raycasting.RaycastPredictor(config['PEDESTRIAN'], dataloader)

    for scan in scans:
        number = int(scan.split('.')[0])
        print(f'sequence: {sequence}, frame: {number}')
        car_pred = rp_car.predict(number)
        ped_pred = rp_ped.predict(number)
        prediction = np.zeros_like(car_pred, dtype=np.uint32)
        for i in range(car_pred.shape[0]):
            if car_pred[i] == ped_pred[i]:
                prediction[i] = car_pred[i]
            elif car_pred[i] == 0:
                prediction[i] = ped_pred[i]
            elif ped_pred[i] == 0:
                prediction[i] = car_pred[i]
            else:
                prediction[i] = 0 # colision, one predicts moving, the other predicts static

        output_name = scan.split('.')[0] + '.label'
        prediction.tofile(os.path.join(predictions, output_name))


if __name__ == '__main__':
    sequences = os.listdir(DATA_PATH)
    sequences.sort()
    sequence = sys.argv[1]
    if sequence not in sequences:
        print(f"wrong sequence {sequence}, found {sequences}")
        exit()
    
    generate_labels_for_sequence(sequence)

        
            

        