import json

from loader.cityscapes_loader import cityscapesLoader
from loader.city_loader import staticLoader
from loader.temporal_loader import temporalLoader
from loader.cityscapes import Cityscapes
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
    }[name]
