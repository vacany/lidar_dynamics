import numpy as np
import os
from tools.ground_removal import *
import time
import copy

EVAL_DIR = 'evaluation_res.txt'
GRID_SEARCH_DIR = 'grid_search.txt'

def evaluate_kitti(config, Predictor_class, predictor_config, verbose=True, file=None):
    sequences = config['KITTI']['SEQUENCES']
    opened_file = False
    if file is None:
        opened_file = True
        file = open(EVAL_DIR, 'w')

    start = time.time()
    for _class in config['CLASSES']:
        mean_precision = 0
        mean_recall = 0
        mean_iou = 0
        num_of_frames = 0
        if verbose:
            print(f"class {_class}")
        file.write(f"class {_class}\n")

        for sequence in sequences:
            sequence_precision = 0
            sequence_recall = 0
            sequence_iou = 0
            if verbose:
                print(f"    sequence: {sequence}")
            file.write(f"    sequence: {sequence}\n")
            gr = Ground_removal(sequence)
            Predictor = Predictor_class(predictor_config[_class], gr)
            frames = sequences[sequence]

            for frame in frames:
                if verbose:
                    print(f"        frame: {frame}", end=' -> ')
                file.write(f"        frame: {frame} -> ")
                frame_num = int(frame)
                prediction = Predictor.predict(frame_num)
                if _class == 'CAR':
                    ground_truth = gr.get_moving_cars_mask(frame_num)
                elif _class == 'PEDESTRIAN':
                    ground_truth = gr.get_moving_pedestrians_mask(frame_num)
                precision, recall, iou = calculate_metrics(prediction, ground_truth)
                if verbose:
                    print(f"precision {np.round(precision * 100,2)}%, recall {np.round(recall * 100,2)}%,"
                        f" iou {np.round(iou * 100,2)}%")
                file.write(f"precision {np.round(precision * 100,2)}%, recall {np.round(recall * 100,2)}%,"
                    f" iou {np.round(iou * 100,2)}%\n")
                sequence_precision += precision; sequence_recall += recall; sequence_iou += iou
                num_of_frames += 1

            if verbose:
                print(f"    average sequence precision {np.round(sequence_precision/len(frames)*100,2)}%, "
                    f"average sequence recall {np.round(sequence_recall/len(frames)*100, 2)}%,"
                    f" average sequence iou {np.round(sequence_iou/len(frames)*100,2)}%\n")
            file.write(f"    average sequence precision {np.round(sequence_precision/len(frames)*100,2)}%, "
                    f"average sequence recall {np.round(sequence_recall/len(frames)*100, 2)}%,"
                    f" average sequence iou {np.round(sequence_iou/len(frames)*100,2)}%\n\n")
            mean_precision += sequence_precision; mean_recall += sequence_recall; mean_iou += sequence_iou
        
        mean_precision /= num_of_frames; mean_recall /= num_of_frames; mean_iou /= num_of_frames
        mean_precision, mean_recall, mean_iou = np.round([mean_precision*100, mean_recall*100, mean_iou*100], 2)
        if verbose:
            print(f"for class {_class} on KITTI dataset, the final average metrics are: precision {mean_precision}%"
                f" recall {mean_recall}% and iou {mean_iou}%")
        file.write(f"for class {_class} on KITTI dataset, the final average metrics are: precision {mean_precision}%"
                f" recall {mean_recall}% and iou {mean_iou}%\n")
        
        end = time.time()
        if verbose:
            print(f'evaluation finished in {np.round(end - start, 1)} seconds')
        file.write(f'evaluation finished in {np.round(end - start, 1)} seconds\n\n')

    if opened_file:
        file.close()
    return [mean_precision, mean_recall, mean_iou]

def recursive_search(x, evaluation_config, Predictor, predictor_config, res_dict, file=None):
    local_copy = copy.deepcopy(x)
    parameters = list(local_copy.keys())
    if parameters == []:
        print(predictor_config['CAR'])
        if file:
            file.write(str(predictor_config['CAR']) + '\n')
        metrics = evaluate_kitti(evaluation_config, Predictor, predictor_config, verbose=False, file=file)
        print(metrics)
        
        res_dict[str(predictor_config['CAR'])] = metrics
        return

    param = parameters[0]
    values = list(local_copy.pop(param))
    for value in values:
        predictor_config['CAR'][param] = value
        #print(f"{param}: {value}")
        recursive_search(local_copy, evaluation_config, Predictor, predictor_config, res_dict, file)


def grid_search(evaluation_config, Predictor_class, predictor_config, parameters):
    '''
    Function for searching the best hyper parameters using brute force all combinations.
    Parameters:
        evaluation_config (dictionary)
        Predictor_class (class)
        predictor_config (dictionary)
        parameters (dictionary): key is name (string) of parameter and value is list of values we want to try out, 
                                    name of parameter must match its name in predictor_config file
    '''
    results = {}
    file = open(GRID_SEARCH_DIR,'w')
    recursive_search(parameters, evaluation_config, Predictor_class, predictor_config, results, file)
    file.write("\n" + "-" * 80 + "\n Results:\n\n")
    for key in results:
        file.write(f"{results[key]}  :  {key}\n\n")
        print(f"{results[key]}  :  {key}\n")
    file.close()



