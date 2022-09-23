from hypernets.core.search_space import ModuleSpace, Choice


class SearchSpaceMixin:

    def __init__(self, **kwargs):
        self.task = kwargs.get('task', None)
        self.timestamp = kwargs.get('timestamp', None)
        self.covariables = kwargs.get('covariables', None)
        self.freq = kwargs.get('freq', None)
        self.metrics = kwargs.get('metrics', None)
        self.window = kwargs.get('window', None)
        self.horizon = kwargs.get('horizon', 1)

    def update_init_params(self, **kwargs):
        if self.task is None and kwargs.get('task') is not None:
            self.task = kwargs.get('task')
        if self.timestamp is None and kwargs.get('timestamp') is not None:
            self.timestamp = kwargs.get('timestamp')
        if self.covariables is None and kwargs.get('covariables') is not None:
            self.covariables = kwargs.get('covariables')
        if self.freq is None and kwargs.get('freq') is not None:
            self.freq = kwargs.get('freq')
        if self.metrics is None and kwargs.get('metrics') is not None:
            self.metrics = kwargs.get('metrics')
        if self.window is None and kwargs.get('window') is not None:
            self.window = kwargs.get('window')
        if self.horizon == 1 and kwargs.get('horizon') is not None:
            self.horizon = kwargs.get('horizon')

    def initial_window_kwargs(self, default_init_kwargs):
        if isinstance(self.window, int):
            default_init_kwargs.update({'window': self.window})
        elif (isinstance(self.window, list) and len(self.window) == 1):
            default_init_kwargs.update({'window': self.window[0]})
        elif (isinstance(self.window, list) and len(self.window) > 1):
            default_init_kwargs.update({'window': Choice(self.window)})
        else:
            raise ValueError('window must be int or list.')
        return default_init_kwargs

class WithinColumnSelector:

    def __init__(self, selector, selected_cols):
        self.selector = selector
        self.selected_cols = selected_cols

    def __call__(self, df):
        intersection = set(df.columns.tolist()).intersection(self.selected_cols)
        if len(intersection) > 0:
            selected_df = df[intersection]
            return self.selector(selected_df)
        else:
            return []


class HyperParams(ModuleSpace):

    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.space.hyperparams = self

    def _compile(self):
        pass

    def _forward(self, inputs):
        return inputs

    def _on_params_ready(self):
        self._compile()