import os, sys, h5py, numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import yaml
from PIL import Image

class LineModDataset(Dataset): 
    def __init__(self, root, split):
        self.num_ppoints = 1024
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])

        self.intrinsic_matrix = np.array([[572.4114, 0.,         325.2611],
                                [0.,        573.57043,  242.04899],
                                [0.,        0.,         1.]])

        self.root = root
        self.cls_dir_list = os.listdir(root)
        self.img_path = []

        for cls_dir in self.cls_dir_list:
            txt_path = os.path.join(self.root, cls_dir, split + '.txt')
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    self.img_path.append(os.path.join(self.root, cls_dir, 'rgb', line.strip() + '.png'))

    def dpt_2_cld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        msk_dp = dpt > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 1:
            return None, None

        dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_mskd = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_mskd = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = dpt_mskd / cam_scale
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate((pt0, pt1, pt2), axis=1)
        return cld, choose

    def __getitem__(self, index):
        path = self.img_path[index]
        cls = path.split('/')[-3]
        item_name = path.split('/')[-1]
        with Image.open(os.path.join(self.root, cls, 'depth', item_name)) as di:
                    dpt = np.array(di)
        with Image.open(os.path.join(self.root, cls, 'mask', item_name)) as li:
            labels = np.array(li)
            labels = (labels > 0).astype("uint8")
        with Image.open(os.path.join(self.root, cls, 'rgb', item_name)) as ri:
            rgb = np.array(ri)[:, :, :3]

        meta_file = open(os.path.join(self.root, cls, 'gt.yml'), "r")
        meta  = yaml.load(meta_file)[int(item_name.split('.png')[0])][0]
        R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        T = np.array(meta['cam_t_m2c']) / 1000.0
        RT = np.concatenate((R, T[:, None]), axis=1)
        rnd_typ = 'real'
        K = self.intrinsic_matrix
        cam_scale = 1000.0

        rgb = rgb[:, :, ::-1].copy()
        msk_dp = dpt > 1e-6
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]

        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
        cld, choose = self.dpt_2_cld(dpt, cam_scale, K)


        labels = labels.flatten()[choose]
        rgb_lst = []
        for ic in range(rgb.shape[0]):
            rgb_lst.append(
                rgb[ic].flatten()[choose].astype(np.float32)
            )
        rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()

        choose = np.array([choose])
        choose_2 = np.array([i for i in range(len(choose[0, :]))])

        cld_rgb = np.concatenate((cld, rgb_pt), axis=1)
        cld_rgb = cld_rgb[choose_2, :]
        cld = cld[choose_2, :]
        labels = labels[choose_2].astype(np.int32).reshape((len(labels), 1))
        rand = np.random.choice(len(cld), self.num_ppoints)
        cld = cld[rand]
        labels = labels[rand]
        #target = np.concatenate((cld, labels), axis=1)
        return cld, labels


    def __len__(self):
        return len(self.img_path)