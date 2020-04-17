from . import (detection, recognition, utils, data_generation, ocr_model, evaluation, datasets,
               custom_objects)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
