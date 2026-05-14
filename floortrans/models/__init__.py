from floortrans.models.hg_furukawa_original import *
from floortrans.models.unet import *
from floortrans.models.segformer import *
from floortrans.models.deeplab import DeepLabV3Plus
from floortrans.models.unet_resnet50 import UNetResNet50


def get_model(name, n_classes=None, version=None):

    if name == 'hg_furukawa_original':
        model = hg_furukawa_original(n_classes=n_classes)
        model.init_weights()

    elif name == 'unet':
        model = UNet(n_classes=n_classes)

    elif name == 'segformer':
        model = segFormer(n_classes=n_classes)
        
    elif name == 'deeplab':
        model = DeepLabV3Plus(n_classes=n_classes)
        
    elif name == 'unet_resnet50':
        model = UNetResNet50(n_classes=n_classes)

    else:
        raise ValueError('Model {} not available'.format(name))

    return model