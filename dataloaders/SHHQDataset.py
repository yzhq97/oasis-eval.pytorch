import os, io
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import math
import joblib
import cv2


class Ceph:

    def __init__(self, bucket, config="petreloss_s_cluster.conf"):

        self.bucket = bucket
        self.client = Client(config)

    def listdir(self, path):

        return list(self.client.list(path))

    def get_bytes(self, path):
        for i in range(10):
            bytes = self.client.get(path)
            if bytes is not None: break
            time.sleep(0.1)
        return bytes

    def get_image(self, path):

        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = self.get_bytes(path)
        if bytes is None: print("error loading {}".format(path))
        img_mem_view = memoryview(bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = Image.fromarray(img_array)

        return img

    def get_numpy_txt(self, path):

        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = self.get_bytes(path)
        if bytes is None: print("error loading {}".format(path))
        return np.loadtxt(io.BytesIO(bytes))

    def get_pkl(self, path, python2=False):
        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = self.get_bytes(path)
        if bytes is None: print("error loading {}".format(path))
        if python2:
            ret = pickle.loads(bytes, encoding="latin1")
        else:
            ret = pickle.loads(bytes)
        return ret

    def put_image(self, path, data):
        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        _, ext = os.path.splitext(path)
        success, array = cv2.imencode("." + ext, data)
        return self.client.put(path, array.tostring())

    def put_pkl(self, path, data):
        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = pickle.dumps(data)
        return self.client.put(path, bytes)


class SHHQDataset(Dataset):

    corrupted = [118464]

    def __init__(self, opt, for_metrics):
        super().__init__()

        self.root = "s3://fbhqv2"
        self.ceph = Ceph(self.root, "petreloss_s_cluster.conf")
        print("SHHQDataset: using ceph {}".format(kwargs["ceph_config"]))

        self.length = opt.dataset_length
        self.aspect_ratio = 2
        self.height = opt.load_size
        self.width = self.height // self.aspect_ratio

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR)])


    def __len__(self):
        return self.length

    def __getitem__(self, index):

        while index in self.corrupted:
            index = (index + 1) % len(self)

        rgb_path = os.path.join(self.root, "images", f"{index + 1:06d}.png")
        mask_path = os.path.join(self.root, "masks", f"{index + 1:06d}.png")
        body_seg_path = os.path.join(self.root, "body_seg", f"{index + 1:06d}.png")

        rgb = np.array(self.ceph.get_image(rgb_path))
        mask = np.array(self.ceph.get_image(mask_path))
        body_segments = np.array(self.ceph.get_image(body_seg_path))

        rgb[mask==0] = 255
        rgb = self.image_transform(rgb)

        # body segments
        body_segments = cv2.resize(body_segments[:, :, 0], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        seg_fg = (body_segments > 0)
        body_segments[seg_fg] += 1  # 0 is reserved for "fake"
        body_segments[~seg_fg] = 1  # 1 is reserved for background

        data = {"image": rgb, "label": body_segments.astype(np.int64), "name": os.path.basename(rgb_path)}

        return data

