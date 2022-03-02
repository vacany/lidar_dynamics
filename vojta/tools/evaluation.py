import numpy as np
import os
from tools.ground_removal import *
import time

OUTPUT_DIR = 'evaluation_res.txt'


def evaluate_kitti(config, Predictor_class, predictor_config):
    sequences = config['KITTI']['SEQUENCES']
    with open(OUTPUT_DIR, 'w+') as file:
        start = time.time()
        for _class in config['CLASSES']:
            mean_precision = 0
            mean_recall = 0
            mean_iou = 0
            num_of_frames = 0
            print(f"class {_class}")
            file.write(f"class {_class}\n")

            for sequence in sequences:
                sequence_precision = 0
                sequence_recall = 0
                sequence_iou = 0
                sequence_num_of_frames = 0
                print(f"    sequence: {sequence}")
                file.write(f"    sequence: {sequence}\n")
                gr = Ground_removal(sequence)
                Predictor = Predictor_class(predictor_config[_class], gr)
                frames = sequences[sequence]

                for frame in frames:
                    print(f"        frame: {frame}", end=' -> ')
                    file.write(f"        frame: {frame} -> ")
                    frame_num = int(frame)
                    prediction = Predictor.predict(frame_num)
                    ground_truth = gr.get_moving_cars_mask(frame_num)
                    precision, recall, iou = calculate_metrics(prediction, ground_truth)
                    print(f"precision {np.round(precision,2)}, recall {np.round(recall,2)}, iou {np.round(iou,2)}")
                    file.write(f"precision {np.round(precision,2)}, recall {np.round(recall,2)}, iou {np.round(iou,2)}\n")
                    sequence_precision += precision; sequence_recall += recall; sequence_iou += iou
                    num_of_frames += 1

                print(f"    average sequence precision {np.round(sequence_precision/len(frames),2)}, "
                    f"average sequence recall {np.round(sequence_recall/len(frames), 2)},"
                    f" average sequence iou {np.round(sequence_iou/len(frames),2)}")
                file.write(f"    average sequence precision {np.round(sequence_precision/len(frames),2)}, "
                    f"average sequence recall {np.round(sequence_recall/len(frames), 2)},"
                    f" average sequence iou {np.round(sequence_iou/len(frames),2)}\n")
                mean_precision += sequence_precision; mean_recall += sequence_recall; mean_iou += sequence_iou
            
            mean_precision /= num_of_frames; mean_recall /= num_of_frames; mean_iou /= num_of_frames
            mean_precision, mean_recall, mean_iou = np.round([mean_precision, mean_recall, mean_iou], 2)
            print(f"for class {_class} on KITTI dataset, the final average metrics are: precision {mean_precision}"
                    f" recall {mean_recall} and iou {mean_iou}")
            file.write(f"for class {_class} on KITTI dataset, the final average metrics are: precision {mean_precision}"
                    f" recall {mean_recall} and iou {mean_iou}\n")
        
        end = time.time()
        print(f'evaluation finished in {np.round(end - start, 2)} seconds')
        file.write(f'evaluation finished in {np.round(end - start, 2)} seconds\n')


