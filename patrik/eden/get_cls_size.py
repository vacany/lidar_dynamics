import json
import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


once_root = '/mnt/beegfs/gpu/temporary/vacekpa2/once/data/'

jsons = glob.glob(once_root + '/*/*.json')

print(jsons)

cyc_list = []
ped_list = []
veh_list = []

for json_file in jsons:
    with open(json_file, 'r') as f:
        anno = json.load(f)

    for i in range(len(anno['frames'])):
        if 'annos' in anno['frames'][i].keys():

            for idx in range(len(anno['frames'][i]['annos']['names'])):

                class_ = anno['frames'][i]['annos']['names'][idx]
                box = anno['frames'][i]['annos']['boxes_3d'][idx]

                if class_ == 'Cyclist':
                    cyc_list.append(box)
                if class_ == 'Pedestrian':
                    ped_list.append(box)
                if class_ == 'Car':
                    veh_list.append(box)
    # break

cyc_array = np.stack(cyc_list)
ped_array = np.stack(ped_list)
veh_array = np.stack(veh_list)


for cls_, array in zip(['veh', 'cyc', 'ped'], [veh_array, cyc_array, ped_array]):
    max_len = np.max(array[:,3])
    min_len = np.min(array[:,3])

    max_wid = np.max(array[:,4])
    min_wid = np.min(array[:,4])

    max_hei = np.max(array[:, 5])
    min_hei = np.min(array[:, 5])

    print(f'Class: {cls_} \n max_len: {max_len:.2f} \t min_len: {min_len:.2f} \n'
          f' max_wid: {max_wid:.2f} \t min_wid: {min_wid:.2f} \n'
          f' max_heg: {max_hei:.2f} \t min_hei: {min_hei:.2f} \n')

fig, a = plt.subplots(3,3)
# a = a.ravel()

classes = ['Vehicle', 'Pedestrian', 'Cyclist']
attri = ['Lenght', 'Width', 'Height']
arrays = [veh_array, ped_array, cyc_array]

for i in range(3):
    for j in range(3):
        ax = a[i, j]

        ax.hist(arrays[j][:, 3 + i], bins=100)
        ax.set_title(f"{classes[j]}")

        ax.set_xlabel(attri[i])
        ax.set_ylabel('N')

plt.tight_layout()
plt.show()
