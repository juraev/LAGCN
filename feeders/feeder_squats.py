import numpy as np
import pickle
import json
import random
import math

from torch.utils.data import Dataset

drop_index = [1, 2, 3, 4, 5, 6, 13, 14, 17, 18]

ner_perm = [0, 1, 3, 5, 7, 2, 4, 6, 8, 9, 11, 13, 15, 17, 10, 12, 14, 16, 18]

def normalize(attempt):
    mean = np.mean(attempt)
    std = np.std(attempt)
    attempt = (attempt - mean) / std
    attempt = attempt.reshape(-1, 29, 3)
    return attempt


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):

        self.squats_root = 'data/squats/'
        self.time_steps = 100
        self.num_joints = 19

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat

        if 'train' in self.label_path:
            self.train_val = 'train'
            file = 's_train.json'
            self.sample_name = ['train_' + str(i) for i in range(4000)]
        else:
            self.train_val = 'val'
            file = 's_test.json'
            self.sample_name = ['test_' + str(i) for i in range(1000)]

        self.load_data(file)
        if normalization:
            self.get_mean_map()


    def load_data(self, file):
        json_file = self.squats_root + file
        with open(json_file) as f:
            content = f.read()
        
        attempts = json.loads(content)

        poses = [json.loads(attempt['pose_info']) for attempt in attempts]
        poses = [[list(pose.values()) for pose in attempt] for attempt in poses]
        
        self.data = [normalize(np.array(attempt)) for attempt in poses]
        
        # drop joints that are not used, delete elements with indices in drop_index
        self.data = [np.delete(attempt, drop_index, axis=1) for attempt in self.data]

        # permute joints to match the order in the graph
        self.data = [attempt[:, ner_perm, :] for attempt in self.data]

        self.label = [int(attempt['feedback_score']) for attempt in attempts]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)*self.repeat

    def __iter__(self):
        return self

    def rand_view_transform(self,X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.label)]
        value = self.data[index % len(self.label)]

        if self.train_val == 'train':
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            value = value + np.random.randn(*value.shape) * 0.02

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1
            scalerValue = np.reshape(scalerValue, (-1, self.num_joints, 3))

            data = np.zeros( (self.time_steps, self.num_joints, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length))*100, self.time_steps)
            random_idx.sort()
            data[:,:,:] = value[random_idx,:,:]
            data[:,:,:] = value[random_idx,:,:]

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1

            scalerValue = np.reshape(scalerValue, (-1, self.num_joints, 3))

            data = np.zeros( (self.time_steps, self.num_joints, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            idx = np.linspace(0,length-1,self.time_steps).astype(np.int)
            data[:,:,:] = value[idx,:,:] # T,V,C

        if 'bone' in self.data_path:
            data_bone = np.zeros_like(data)
            for bone_idx in range(17):
                data_bone[:, self.bone[bone_idx][0] - 1, :] = data[:, self.bone[bone_idx][0] - 1, :] - data[:, self.bone[bone_idx][1] - 1, :]
            data_bone[:, 2, :] = data[:, 2, :]
            data = data_bone

        ## for joint modality
        ## separate trajectory from relative coordinate to each frame's spine center
        else:
            # # there's a freedom to choose the direction of local coordinate axes!
            trajectory = data[:, 2]
            # let spine of each frame be the joint coordinate center
            data = data - data[:, 2:3]
            #
            # ## works well with bone, but has negative effect with joint and distance gate
            data[:, 2] = trajectory

        if 'motion' in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion
        data = np.transpose(data, (2, 0, 1))
        C,T,V = data.shape
        data = np.reshape(data,(C,T,V,1))

        return data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
