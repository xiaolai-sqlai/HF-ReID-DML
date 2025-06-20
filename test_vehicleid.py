# -*- coding: utf-8 -*-
from __future__ import print_function, division

from cv2_transform import transforms
from torch.utils.data import DataLoader
import torch

from network.hfnet import HFNet
from data_read import ImageTxtDataset
from train import str2bool

import argparse, time, os, sys, random
import numpy as np
from os import path as osp

def get_data(batch_size, test_set, query_set):
    transform_test = transforms.Compose([
        transforms.Resize(size=(opt.img_height, opt.img_width)),
        transforms.ToTensor(),
    ])

    test_imgs = ImageTxtDataset(test_set, transform=transform_test)
    query_imgs = ImageTxtDataset(query_set, transform=transform_test)

    test_data = DataLoader(test_imgs, batch_size, shuffle=False, num_workers=8)
    query_data = DataLoader(query_imgs, batch_size, shuffle=False, num_workers=8)
    return test_data, query_data


def extract_feature(net, dataloaders):
    count = 0
    features = []
    for img, _ in dataloaders:
        n = img.shape[0]
        count += n
        ff = np.zeros((n, opt.feat_num*len(opt.decoder.split("_"))), dtype=np.float32)
        for i in range(2):
            if(i==1):
                img = torch.flip(img, [3])
            with torch.no_grad():
                f = torch.cat(net(img.cuda()), dim=1).detach().cpu().numpy()
            ff = ff+f
        features.append(ff)
    features = np.concatenate(features)
    features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))
    return features

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--img-height', type=int, default=384)
    parser.add_argument('--img-width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dataset-root', type=str, default="../dataset/")
    parser.add_argument('--net', type=str, default="regnet_y_3_2gf", help="regnet_y_3_2gf, regnet_y_8gf")
    parser.add_argument('--decoder', type=str, default="rpm_max")
    parser.add_argument('--gpus', type=str, default="0,1", help='number of gpus to use.')

    opt = parser.parse_args()

    data_dir = osp.join(opt.dataset_root, "VehicleID_V1.0")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    label_to_items = {}
    for line in open(osp.join(data_dir, "train_test_split", "test_list_800.txt")).readlines():
        name, label = line.strip().split()
        if label not in label_to_items.keys():
            label_to_items[label] = []
            
        label_to_items[label].append([
            osp.join(data_dir, 'image', name+".jpg"),
            label, 
            name
        ])
    
    test_set = []
    test_label = []
    test_cam = []
    query_set = []
    query_label = []
    query_cam = []

    for label in label_to_items.keys():
        items = label_to_items[label]
        random.shuffle(items)
        for i, item in enumerate(items):
            if i == 0:
                test_set.append(item[:2])
                test_label.append(item[1])
                test_cam.append(item[2])
            else:
                query_set.append(item[:2])
                query_label.append(item[1])
                query_cam.append(item[2])
    test_label = np.array(test_label)
    test_cam = np.array(test_cam)
    query_label = np.array(query_label)
    query_cam = np.array(query_cam)

    ######################################################################
    # Load Collected data Trained model
    mod_pth = osp.join('params', 'ema.pth')

    net = HFNet(num_classes=13164, net=opt.net, decoder=opt.decoder)
    opt.feat_num = net.feat_num

    net.load_state_dict(torch.load(mod_pth), False)
    net.cuda()
    net.eval()

    # Extract feature
    test_loader, query_loader = get_data(opt.batch_size, test_set, query_set)
    print('start test')
    test_feature = extract_feature(net, test_loader)
    print('start query')
    query_feature = extract_feature(net, query_loader)

    num = query_label.size
    dist_all = np.dot(query_feature, test_feature.T)

    CMC = np.zeros(test_label.size)
    ap = 0.0
    for i in range(num):
        cam = query_cam[i]
        label = query_label[i]
        index = np.argsort(-dist_all[i])

        query_index = np.argwhere(test_label==label)
        camera_index = np.argwhere(test_cam==cam)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index = np.intersect1d(query_index, camera_index)
    
        ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC/num #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/num))

