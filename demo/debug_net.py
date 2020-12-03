import os
import time

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from mmseg.models.backbones import BaseNet

warm_step = 20
test_step = 20
img_size_list = [(1, 3, 1024, 2048)]


def print_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


def benchmark(net, img_size):
    device = torch.device('cuda')
    model = net.to(device)
    model.eval()
    img = torch.empty(size=img_size, device=device)
    with torch.no_grad():

        for _ in range(warm_step):
            _ = model(img)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(test_step):
            _ = model(img)

        torch.cuda.synchronize()
        end = time.perf_counter()
    return test_step / (end - start)


def model_test():
    name = 'gluon_resnet18_v1b'
    model = BaseNet('gluon_resnet18_v1b')
    img = torch.randn((1, 3, 1024, 2048))
    ret = model(img)
    for x in ret:
        print(x.shape)
    for img_size in img_size_list:
        fps = benchmark(model, img_size)
        print('model: ', name, 'img_size: ', img_size, 'fps:', round(fps, 1))


def find_shape():
    img_path = '/home/jpl/data/pycode/mmsegmentation/data/Vistas/training/images'
    h, w = 0, 0
    for path in tqdm(os.listdir(img_path)):
        whole_path = os.path.join(img_path, path)
        img: Image.Image = Image.open(whole_path)
        h += img.size[0]
        w += img.size[1]
    print(h / 18000)  # 3418.662888888889
    print(w / 18000)  # 2481.146277777778


def find_sto():
    img_path = '/home/jpl/data/pycode/mmsegmentation/data/Vistas/training/images'
    num_pixels = 3418.662888888889 * 2481.146277777778 * 18000
    # color = np.zeros(3)
    # for path in tqdm(os.listdir(img_path)):
    #     whole_path = os.path.join(img_path, path)
    #     img = cv.imread(whole_path)
    #     for i in range(3):
    #         color[2 - i] += np.sum(img[:, :, i])
    # mean = color / num_pixels
    mean = np.array([80.5423, 91.3162, 81.4312])

    print(mean)

    color = np.zeros(3)
    for path in tqdm(os.listdir(img_path)):
        whole_path = os.path.join(img_path, path)
        img = cv.imread(whole_path)
        for i in range(3):
            color[2 - i] += np.sum((img[:, :, i] - mean[i]) ** 2)
    var = color / num_pixels

    print(np.sqrt(var))


if __name__ == '__main__':
    find_sto()
