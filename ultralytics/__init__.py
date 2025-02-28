

__version__ = "YOLO:8.1.8"

from ultralytics.models import YOLO
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "download",
    "settings",
)
