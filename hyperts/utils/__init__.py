from . import consts
from . import toolbox
from . import metrics
from . import transformers
from hyperts.utils.toolbox import register_tstoolbox, TSToolBox

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