import numpy as np
import pyransac3d as pyrsc
from data.point_clouds.clustering import structure
from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset, get_ego_bbox

dataset = SemKittiDataset()
batch = dataset.get_multi_frame(460)

points = batch['points'][batch['points'][:,3] > 0.98]

# points = load_points(.) # Load your point cloud as a numpy array (N, 3)



model = structure.DBSCAN(0.9, min_samples=3)
model.fit(points[:,:3])
label = model.labels_

import numpy as np
import matplotlib.pyplot as plt

from data.point_clouds.box import get_bbox_points

fig = plt.figure()
ax = fig.gca(projection='3d')
bbox_list = []
for i in range(np.max(label)):
    if i == 0: continue
    cluster = points[label == i]
    orig_center = cluster[:,:3].mean(0)
    cluster[:,:3] -= cluster[:,:3].mean(0)

    center = cluster[:,:3].mean(0)

    plane1 = pyrsc.Plane()

    best_eq, best_inliers = plane1.fit(cluster[:,:3], thresh=0.1, minPoints=len(cluster), maxIteration=100)

    print(best_eq, best_inliers)

    a,b,c,d = best_eq

    yaw = np.arctan2(b, a)

    _, _, _, l, w, h, _ = get_ego_bbox()

    bbox = np.array(([center[0] + l/2, center[1] + w/2, center[2], l, w, h, yaw]))[None,:]
    bbox[:,:3] += orig_center

    corners=get_bbox_points(bbox)


    x = np.linspace(-1,1,2)
    y = np.linspace(-1,1,2)

    X,Y = np.meshgrid(x,y)
    Z = (d - a*X - b*Y) / c




    # surf = ax.plot_surface(X, Y, Z, alpha=0.2)


    ax.scatter(cluster[:,0], cluster[:,1], cluster[:,2], color='green')
    ax.scatter((0,a),(0,b),(0,c), color='red')
    ax.scatter(corners[:,0], corners[:,1], corners[:,2], color='blue')

    bbox_list.append(corners)
    break
plt.show()

bboxes = np.concatenate(bbox_list)
points = np.concatenate((points[:,:3], bboxes[:,:3]))
labels = np.concatenate((label, bboxes[:,3]))

import pptk
v=pptk.viewer(points[:,:3], label)
v.set(point_size=0.05)

