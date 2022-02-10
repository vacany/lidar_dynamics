import os
import yaml
import socket

# Experiment folder
def default_root_dir():
    server = socket.gethostname()
    if 'rci' in server or len(server) == 3:
        root_dir = '/mnt/beegfs/gpu/temporary/vacekpa2/experiments/'

    elif server == 'Patrik':
        root_dir = '/home/patrik/mnt/hdd/iros_2022/'

    # elif server == 'Goedel':

    # elif server == 'vojta':

    return root_dir

# Tree structure
def exp_structure(exp_root, sequence, config):

    exp_root = exp_root
    config_file = exp_root + '/config.yaml'
    images = exp_root + '/images'
    gen_labels = exp_root + f'/data/{sequence}/gen_labels'

    for folder in [exp_root, images, gen_labels]:
        os.makedirs(folder, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(config, f)

def check_valid_experiments(config):
    #mainly dataset? Maybe not necessary
    return
