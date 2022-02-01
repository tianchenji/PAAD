import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.collections as mcoll

from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    'font.size': 26
})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['axes.facecolor'] = 'black'

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates.
    Output shape: numlines x (points per line) x 2 (x and y) array
    """

    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(x, y, z, linewidth=5):

    segments = make_segments(x, y)
    
    top    = cm.get_cmap('summer', 128)
    bottom = cm.get_cmap('autumn_r', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='GreenRed')
    lc     = mcoll.LineCollection(segments, array=z, cmap=newcmp,
                                norm=plt.Normalize(0.0, 1.0), linewidth=linewidth)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def get_pred_traj_ego(pred_Zs, pred_Xs):

    # focal length in pixels
    f = 460
    # camera height in meters
    Y = - 0.23
    # image size in pixels
    w_ego, h_ego = 320, 240

    # project 3D points onto the image plane
    xs = f * pred_Xs / pred_Zs
    ys = f * Y / pred_Zs

    # convert 2D points into the image coordinate
    xs = - xs + w_ego / 2
    ys = - ys + h_ego / 2

    xs, ys = xs.astype(int), ys.astype(int)

    pred_traj = [(x, y) for x, y in zip(xs, ys)]

    return np.array(pred_traj)

def datapoint_vis(image_path, csv_path, data_idx):

    map_float = lambda x: np.array(list(map(float, x)))
    map_int   = lambda x: np.array(list(map(int, x)))

    df = pd.read_csv(csv_path)

    # visualize the image
    img_name    = os.path.join(image_path, df['image_name'][data_idx])
    camera_view = mpimg.imread(img_name)

    plt.figure(figsize=(12, 12))
    plt.imshow(camera_view)
    plt.axis([-0.5, 319.5, 239.5, -0.5])
    plt.xticks([])
    plt.yticks([])

    # visualize the planned trajectory
    pred_Zs   = df['pred_traj_x'][data_idx][1:-1].split(',')
    pred_Xs   = df['pred_traj_y'][data_idx][1:-1].split(',')
    pred_Zs   = map_float(pred_Zs)
    pred_Xs   = map_float(pred_Xs)
    pred_traj = get_pred_traj_ego(pred_Zs, pred_Xs)

    p_of_failure = df['label'][data_idx][1:-1].split(',')
    p_of_failure = map_float(p_of_failure)[1:]

    lc = colorline(pred_traj[:, 0], pred_traj[:, 1], p_of_failure)

    ax      = plt.gca()
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("left", size="5%", pad=0.3)

    plt.colorbar(lc, cax=cax, label='Predicted Probability of Failure')
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")

    # visualize the point at which the failure is detected
    if max(p_of_failure) >= 0.5:
        time_of_failure = list(p_of_failure >= 0.5).index(1)
        ax.plot(pred_traj[time_of_failure, 0], pred_traj[time_of_failure, 1],
                    color='red', marker='X', markersize=25)

    # plot lidar point cloud
    theta = np.linspace(-0.25*np.pi, 1.25*np.pi, 1081)

    lidar_polar = df['lidar_scan'][data_idx][1:-1].split(',')
    lidar_polar = map_float(lidar_polar)

    lidar_x = lidar_polar * np.cos(theta)
    lidar_y = lidar_polar * np.sin(theta)

    plt.figure(figsize=(8, 11))

    plt.plot(lidar_x, lidar_y, ls='None', color='white', marker='.', markersize=5)
    plt.plot(0, 0, color='tab:blue', marker='^', markersize=25)
    plt.axis([-1, 1, -0.75, 2.0])
    plt.xticks([])
    plt.yticks([])

    plt.show()

if __name__ == '__main__':
    train_image_path = 'train_set/images_train'
    train_csv_path   = 'train_set/labeled_data_train.csv'
    test_image_path  = 'test_set/images_test'
    test_csv_path    = 'test_set/labeled_data_test.csv'

    data_idx = 98

    datapoint_vis(test_image_path, test_csv_path, data_idx)