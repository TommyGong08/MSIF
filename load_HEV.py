import os
import pickle
import re
import numpy as np
from tqdm import tqdm

train_dir = './datasets/SourceFile/train/'
val_dir = './datasets/SourceFile/val/'
train_save_dir = './datasets/HEV/train'
val_save_dir = './datasets/HEV/val'
flow_train_dir = './datasets/HEV/flow_train'
flow_val_dir = './datasets/HEV/flow_val'
flow_train_filename = 'flow_train.pkl'
flow_val_filename = 'flow_val.pkl'
H = 720
W = 1280


def load_HEV_data(data_dir):
    all_file = os.listdir(data_dir)
    data = {}
    video_index = []
    flow_dict = {}

    for track_file in all_file:
        searchObj = re.match('(.*)_(.*).pkl', track_file)

        VideoName = int(searchObj.group(1))
        video_index.append(VideoName)
        OjbID = int(searchObj.group(2))

        with open(data_dir + track_file, 'rb') as f:
            track = pickle.load(f)
        bbox = track['bbox'][0:len(track['frame_id'])]
        flow = track['flow'][0:len(track['frame_id'])]

        if flow_dict.__contains__(VideoName):
            flow_dict[VideoName][OjbID] = {}
            flow_dict[VideoName][OjbID]['flow'] = flow
            flow_dict[VideoName][OjbID]['frame_id'] = track['frame_id']
        else:
            flow_dict[VideoName] = {}
            flow_dict[VideoName][OjbID] = {}
            flow_dict[VideoName][OjbID]['flow'] = flow
            flow_dict[VideoName][OjbID]['frame_id'] = track['frame_id']

        bbox[:, 0] = bbox[:, 0] * W
        bbox[:, 1] = bbox[:, 1] * H

        frame_id = track['frame_id']

        OjbID_list = np.array([OjbID] * len(track['frame_id']))

        output = np.vstack((frame_id, OjbID_list))
        output = np.hstack((output.T, bbox[:, 0:2]))
        if data.__contains__(VideoName):
            data[VideoName] = np.vstack((data[VideoName], output))
        else:
            data[VideoName] = output

    return data, flow_dict


def sort_and_save(data, save_dir):
    for VideoName, all_frame in data.items():
        all_frame = all_frame[all_frame[:, 0].argsort()]
        filename = str(VideoName) + '.txt'
        path = os.path.join(save_dir, filename)
        np.savetxt(path, all_frame, fmt='%.04f')


def save_flow(flow_dict, flow_save_path, filename):
    path = os.path.join(flow_save_path, filename)
    with open(path, 'wb') as f:
        pickle.dump(flow_dict, f)


train_data, flow_train = load_HEV_data(train_dir)
save_flow(flow_train, flow_train_dir, flow_train_filename)
sort_and_save(train_data, train_save_dir)

# val_data, flow_val = load_HEV_data(val_dir)
# save_flow(flow_val, flow_val_dir, flow_val_filename)
# sort_and_save(val_data, val_save_dir)
