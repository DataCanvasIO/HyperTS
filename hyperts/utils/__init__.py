from . import consts
from . import tstoolbox
from . import metrics
from . import transformers
from .tstoolbox import TSToolBox
from ._base import register_tstoolbox

register_tstoolbox(TSToolBox, None)

try:
    from .dask_ex._ts_toolbox import TSDaskToolBox
    register_tstoolbox(TSDaskToolBox, None)
except ImportError:
    print("Import TSDaskToolBox Error")

try:
    from .cuml_ex._ts_toolbox import TSCumlToolBox
    register_tstoolbox(TSCumlToolBox, None)
except ImportError:
    print("Import TSCumlToolBox Error")