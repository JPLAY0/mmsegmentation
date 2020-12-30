_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# model settings
num_classes = 19
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
        channels=96,
        input_transform='multiple_select',
        dropout_ratio=-1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='LovaszLoss', loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            in_index=3,
            channels=128,
            num_convs=2,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='LovaszLoss', loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            in_index=2,
            channels=128,
            num_convs=2,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='LovaszLoss', loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            in_index=1,
            channels=128,
            num_convs=2,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='LovaszLoss', loss_weight=0.4)),

    ]
)
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
