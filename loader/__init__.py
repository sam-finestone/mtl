import json

from loader.cityscapes_loader import cityscapesLoader
from loader.city_loader import cityscapesLoader2
from loader.temporal_loader import temporal_loader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
    }[name]
