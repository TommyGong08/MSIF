import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# import scipy.ndimage
# from tqdm import tqdm
import pickle
import os

ngridx = 200
ngridy = 100
# H = 720
# W = 1280
H = 1200
W = 1920


fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
color = 'Greens'
datasets = './checkpoint'
raw_track_name = 'your-experiment-name/raw_data_dic.pickle'
lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario


def plot_one_frame(car_num, color, track):
    obs_track = []
    trgt_track = []
    pred_track = []

    # convert track into plot-friendly format
    # format is: [car_num,[x, y]]
    for frame in track['obs']:
        # frame[car_num, 0] = frame[car_num, 0] * 1280
        # frame[car_num, 1] = frame[car_num, 1] * 640
        obs_track.append(frame[car_num, :])
    for frame in track['trgt']:
        # frame[car_num, 0] = frame[car_num, 0] * 1280
        # frame[car_num, 1] = frame[car_num, 1] * 640
        trgt_track.append(frame[car_num, :])
    for index, track in enumerate(track['pred']):
        # print('index:',index)
        tmp_track = []
        for frame in track:
            # frame[car_num, 0] = frame[car_num, 0] * 1280
            # frame[car_num, 1] = frame[car_num, 1] * 640
            tmp_track.append(frame[car_num, :])
        pred_track.append(np.array(tmp_track))

    obs_track = np.array(obs_track)
    trgt_track = np.array(trgt_track)
    pred_track_sample = np.array(pred_track)[np.random.choice(len(pred_track), size=1, replace=False)]

    # to plot a contour, there needs to add an z axis
    # if prediction gives 12 frames, then the z axis will be: 12,11,10,...,3,2,1.
    for index, track in enumerate(pred_track):
        z = np.linspace(len(track), 1, len(track))
        z = z[:, np.newaxis]
        pred_track[index] = np.hstack((track, z))

    # get all prediction track together
    pred_track_all = pred_track[0]
    for index, frame in enumerate(pred_track):
        if index == 0:
            continue
        pred_track_all = np.vstack((pred_track_all, frame))

    # convert track into grids
    x = pred_track_all[:, 0]
    y = pred_track_all[:, 1]
    z = pred_track_all[:, 2]

    xi = np.linspace(min(x), max(x), ngridx)
    yi = np.linspace(min(y), max(y), ngridy)

    Xi, Yi = np.meshgrid(xi, yi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)

    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    cntr = ax.contourf(xi, yi, zi, levels=5, cmap=color)

    # real_track = np.vstack((obs_track, trgt_track))

    ax.plot(obs_track[:, 0], obs_track[:, 1], 'b-.', linewidth=0.5)
    ax.plot(trgt_track[:, 0], trgt_track[:, 1], 'r-.', linewidth=0.5)
    ax.set_ylim(0, H)
    ax.set_xlim(0, W)
    for track in pred_track_sample:
        ax.plot(track[:, 0], track[:, 1], 'g-.', linewidth=0.5)

    return


data_path = os.path.join(datasets, raw_track_name)

with open(data_path, 'rb') as file:
    raw_data_dic_ = pickle.load(file)

# open interactive mode
# plt.ion()

for frame_dic, track in raw_data_dic_.items():
    # if frame_dic%5 != 0:
    #     continue
    print('current frame is:', frame_dic)
    ax.cla()
    ax.set_title('frame:%d' % frame_dic)

    for car_num in range(track['obs'].shape[1]):
        plot_one_frame(car_num, color, track)

    plt.pause(0.01)

# close interactive mode
# plt.ioff()
plt.show()
