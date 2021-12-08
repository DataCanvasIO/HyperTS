import os

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

class suppress_stdout_stderr(object):
    ''' Suppressing Stan optimizer printing in Prophet Wrapper.
        A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    References
    ----------
    https://github.com/facebook/prophet/issues/223
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)