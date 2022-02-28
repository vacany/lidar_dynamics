from mayavi import mlab


def plot_bounding_box(pts, fig):
    '''
    Parameters:
        pts (numpy Nx3 array): point cloud
        fig (mayavi.mlab figure): figure to plot to
    Returns:
        None
    '''

    bb_x_min =  pts[:,0].min()
    bb_x_max =  pts[:,0].max()

    bb_y_min =  pts[:,1].min()
    bb_y_max =  pts[:,1].max()

    bb_z_min =  pts[:,2].min()
    bb_z_max =  pts[:,2].max()

    # along x axis
    mlab.plot3d([bb_x_min, bb_x_max], [bb_y_min, bb_y_min], [bb_z_min, bb_z_min], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_min, bb_x_max], [bb_y_max, bb_y_max], [bb_z_min, bb_z_min], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_min, bb_x_max], [bb_y_min, bb_y_min], [bb_z_max, bb_z_max], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_min, bb_x_max], [bb_y_max, bb_y_max], [bb_z_max, bb_z_max], color=(1,0.7,0.3), figure=fig)

    # along y axis
    mlab.plot3d([bb_x_min, bb_x_min], [bb_y_min, bb_y_max], [bb_z_min, bb_z_min], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_min, bb_x_min], [bb_y_min, bb_y_max], [bb_z_max, bb_z_max], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_max, bb_x_max], [bb_y_min, bb_y_max], [bb_z_min, bb_z_min], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_max, bb_x_max], [bb_y_min, bb_y_max], [bb_z_max, bb_z_max], color=(1,0.7,0.3), figure=fig)

    # along z axis
    mlab.plot3d([bb_x_min, bb_x_min], [bb_y_min, bb_y_min], [bb_z_min, bb_z_max], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_min, bb_x_min], [bb_y_max, bb_y_max], [bb_z_min, bb_z_max], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_max, bb_x_max], [bb_y_min, bb_y_min], [bb_z_min, bb_z_max], color=(1,0.7,0.3), figure=fig)
    mlab.plot3d([bb_x_max, bb_x_max], [bb_y_max, bb_y_max], [bb_z_min, bb_z_max], color=(1,0.7,0.3), figure=fig)
