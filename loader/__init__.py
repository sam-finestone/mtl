import json

from loader.cityscapes_loader import cityscapesLoader
from loader.static_loader import staticLoader
from loader.temporal_loader import temporalLoader
from loader.cityscapes import Cityscapes
from loader.temporal_loader_3 import temporalLoader3
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
    }[name]
