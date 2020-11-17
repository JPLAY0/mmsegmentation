import time

import torch

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


if __name__ == '__main__':
    name = 'gluon_resnet18_v1b'
    model = BaseNet('gluon_resnet18_v1b')
    img = torch.randn((1, 3, 1024, 2048))
    ret = model(img)
    for x in ret:
        print(x.shape)
    for img_size in img_size_list:
        fps = benchmark(model, img_size)
        print('model: ', name, 'img_size: ', img_size, 'fps:', round(fps, 1))
