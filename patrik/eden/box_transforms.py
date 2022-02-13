import numpy as np
import math
import matplotlib.pyplot as plt

def get_angle_between_points(pt1, pt2, show=False):
    pt1 = pt1 - pt1
    pt2 = pt2 - pt1

    angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

    if show:
        print(angle)
        plt.plot(pt2[0], pt2[1], 'r*')
        plt.plot(pt1[0], pt1[1], 'b*')

        plt.show()

