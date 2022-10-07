import os
import math
import sys
import re
import pickle
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time


def image_format_convert(image_file_name):
    image = cv2.imread(image_file_name)
    image = image.copy()
    image = cv2.resize(image, (480, 300), interpolation=cv2.INTER_CUBIC)
    image = image.transpose((2, 0, 1))
    return image


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    # V shape:(sequance length, nodes num, feature num)
    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, flow_dir, image_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='space', norm_lap_matr=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.flow_dir = flow_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        # load trajectories
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        flow_list = []
        image_list = []
        loss_mask_list = []
        non_linear_ped = []

        all_flows = os.listdir(self.flow_dir)
        all_flows = [os.path.join(self.flow_dir, _path) for _path in all_flows]

        for path in all_flows:
            with open(path, 'rb') as f:
                self.flow = pickle.load(f)

        for path in all_files:

            searchObj = re.match('(.*)/(.*).txt', path)
            VideoName = int(searchObj.group(2))
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                # print('*'*30)
                # print('current pedestrian index:',peds_in_curr_seq)
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                curr_flow = []
                curr_image = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)  # format: [frame_id, ped_id, x, y]
                    # pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    # pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    # print("ped 1:",curr_ped_seq)

                    if curr_ped_seq.shape[0] != self.seq_len:
                        continue
                    # print('ped id:', ped_id)
                    curr_frame_id = curr_ped_seq[0, 0]
                    # print('frame id:', curr_frame_id)
                    ped_flow_all = self.flow[VideoName][ped_id]
                    flow_index = np.where(ped_flow_all['frame_id'] == curr_frame_id)[0][0]
                    curr_ped_flow = ped_flow_all['flow'][flow_index:flow_index + self.seq_len]
                    curr_flow.append(curr_ped_flow)
                    # print("############")
                    # print(curr_ped_flow.shape)  # (20, 5, 5, 2)

                    # image
                    curr_ped_images = []
                    for i in range(curr_ped_seq.shape[0]):
                        image_frame_id = str(int(curr_ped_seq[i][0])+1).zfill(6)
                        image_file_name = image_dir + str(VideoName) + '/' + str(image_frame_id) + '.jpg'
                        # image_output = image_format_convert(image_file_name)
                        curr_ped_images.append(image_file_name)  # curr_image: [20, 3, 300, 480]

                    # curr_ped_images = np.array(curr_ped_images)
                    curr_image.append(curr_ped_images)

                    # Make coordinates relative
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    # print("ped 2:",curr_ped_seq)
                    _idx = num_peds_considered
                    # print("curr_seq:",curr_seq)
                    curr_seq[_idx, :, :] = curr_ped_seq
                    curr_seq_rel[_idx, :, :] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, :] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    curr_flow = np.array(curr_flow, dtype="float64")
                    # print(curr_flow.shape) # (2, 20, 5, 5, 2)
                    flow_list.append(curr_flow)
                    # print(len(flow_list))
                    # curr_image = np.array(curr_image, dtype="int8")
                    # print(curr_image.shape)  # (2, 20, 3, 300, 480)
                    image_list.append(curr_image)
                    # print(len(image_list))

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        flow_list = np.concatenate(flow_list, axis=0)
        image_list = np.concatenate(image_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.obs_flow = torch.from_numpy(
            flow_list[:, :self.obs_len, :, :, :]).type(torch.float)
        self.pred_flow = torch.from_numpy(
            flow_list[:, self.obs_len:, :, :, :]).type(torch.float)
        self.obs_image = image_list[:, :self.obs_len]
        self.pred_image = image_list[:, self.obs_len:]
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]

            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        obs_image = []
        for i in range(start, end):
            curr_ped_image = []
            for j in range(self.obs_image.shape[1]):
                image = image_format_convert(self.obs_image[i][j])
                curr_ped_image.append(image)
            obs_image.append(curr_ped_image)
        obs_image = np.array(obs_image)
        self.obs_image_final = torch.from_numpy(obs_image).type(torch.float)

        pred_image = []
        for i in range(start, end):
            curr_ped_image = []
            for j in range(self.pred_image.shape[1]):
                image = image_format_convert(self.pred_image[i, j])
                curr_ped_image.append(image)
        pred_image.append(curr_ped_image)
        pred_image = np.array(pred_image)
        self.pred_image_final = torch.from_numpy(pred_image).type(torch.float)

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.obs_flow[start:end, :], self.pred_flow[start:end, :],
            # self.obs_image[start:end, :], self.pred_image[start:end, :],
            self.obs_image_final, self.pred_image_final,
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out
