_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='checkpoints/resnet18-5c106cde.pth',
    backbone=dict(
        type='ResNet18',
    ),
    decode_head=dict(
        type='SINGLEHead',
        in_channels=[64, 128, 256, 512],
        in_index=(0, 1, 2, 3),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=-1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
