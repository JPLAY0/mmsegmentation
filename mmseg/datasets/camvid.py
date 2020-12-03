from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamVidDataset(CustomDataset):
    CLASSES = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian',
               'Bicyclist', 'Unlabelled']
    PALETTE = [[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [60, 40, 222], [128, 128, 0],
               [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(CamVidDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
