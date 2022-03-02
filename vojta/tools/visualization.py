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


def plot_prediction_cars(dataloader, predictor, frame, figure):
    pts, intens = dataloader.get_frame(frame)
    ground_truth = dataloader.get_moving_cars_mask(frame)
    prediction = predictor.predict(frame)

    TP = ground_truth & prediction
    FN = ground_truth & ~prediction
    FP = ~ground_truth & prediction
    rest = ~ground_truth & ~prediction

    if TP.sum() > 0:
        mlab.points3d(pts[:,0][TP], pts[:,1][TP],
                    pts[:,2][TP], color=(0,1,0), mode='point', figure=figure)
    if FP.sum() > 0:
        mlab.points3d(pts[:,0][FP], pts[:,1][FP], pts[:,2][FP], color=(0,0,1), mode='point', figure=figure)

    if FN.sum() > 0:  
        mlab.points3d(pts[:,0][FN], pts[:,1][FN], pts[:,2][FN], color=(1,0,0), mode='point', figure=figure)
    mlab.points3d(pts[:,0][rest], pts[:,1][rest], pts[:,2][rest], color=(1,1,1), mode='point', figure=figure)