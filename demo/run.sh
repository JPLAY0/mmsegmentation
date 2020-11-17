#!/bin/bash

source /usr/local/anaconda/bin/activate mm
cd /data/jpl/pycodes/mmsegmentation
python tools/train.py configs/me/sg_resnet18_cityscapes.py


# ps aux|grep run.sh