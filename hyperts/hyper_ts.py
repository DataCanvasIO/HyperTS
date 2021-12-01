import pickle

import numpy as np
from sklearn import pipeline as sk_pipeline

from hypernets.utils import fs, logging
from hypernets.tabular import get_tool_box
from hypernets.tabular.metrics import calc_score
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.core.meta_learner import MetaLearner
from hypernets.pipeline.base import ComposeTransformer
from hypernets.dispatchers.in_process_dispatcher import InProcessDispatcher

from hyperts.utils import consts

logger = logging.get_logger(__name__)


class HyperTSEstimator(Estimator):
    def __init__(self, task, space_sample, data_cleaner_params=None):
        super(HyperTSEstimator, self).__init__(space_sample=space_sample, task=task)
        self.data_pipeline = None
        self.data_cleaner_params = data_cleaner_params
        self.model = None  # Time-Series model
        self.cv_models_ = None
        self.data_cleaner = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self.pos_label = None
        self.transients_ = {}

        self._build_model(space_sample)

    def _build_model(self, space_sample):
        space, _ = space_sample.compile_and_forward()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        if outputs[0].estimator is None:
            outputs[0].build_estimator(self.task)
        self.model = outputs[0].estimator
        self.fit_kwargs = outputs[0].fit_kwargs

        pipeline_module = space.get_inputs(outputs[0])
        assert len(pipeline_module) == 1, 'The `HyperEstimator` can only contains 1 input.'
        if isinstance(pipeline_module[0], ComposeTransformer):
            self.data_pipeline = self.build_pipeline(space, pipeline_module[0])

    def build_pipeline(self, space, last_transformer):
        transformers = []
        while True:
            next, (name, p) = last_transformer.compose()
            transformers.insert(0, (name, p))
            inputs = space.get_inputs(next)
            if inputs == space.get_inputs():
                break
            assert len(inputs) == 1, 'The `ComposeTransformer` can only contains 1 input.'
            assert isinstance(inputs[0], ComposeTransformer), \
                'The upstream node of `ComposeTransformer` must be `ComposeTransformer`.'
            last_transformer = inputs[0]
        assert len(transformers) > 0
        if len(transformers) == 1:
            return transformers[0][1]
        else:
            pipeline = sk_pipeline.Pipeline(steps=transformers)
            return pipeline

    def summary(self):
        if self.data_pipeline is not None:
            return f"{self.data_pipeline.__repr__(1000000)}"
        else:
            return "HyperTSEstimator"

    def fit_cross_validation(self, X, y, verbose=0, stratified=True, num_folds=3, pos_label=None,
                             shuffle=False, random_state=9527, metrics=None, **kwargs):
        return None, None, None

    def get_iteration_scores(self):
        return None

    def fit(self, X, y, pos_label=None, verbose=0, **kwargs):
        if self.data_pipeline is not None:
            X_transformed = self.data_pipeline.fit_transform(X)
        else:
            X_transformed = X
        self.model.fit(X_transformed, y)

    def predict(self, X, verbose=0, **kwargs):
        if self.data_pipeline is not None:
            X_transformed = self.data_pipeline.transform(X)
        else:
            X_transformed = X
        preds = self.model.predict(X_transformed)
        return preds

    def predict_proba(self, X, verbose=0, **kwargs):
        return None

    def evaluate(self, X, y, metrics=None, verbose=0, **kwargs):
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        if self.data_pipeline is not None:
            X_transformed = self.data_pipeline.transform(X)
        else:
            X_transformed = X
        if metrics is None:
            if self.task in [consts.TASK_FORECAST, consts.TASK_UNIVARIABLE_FORECAST,
                                consts.TASK_MULTIVARIABLE_FORECAST, consts.TASK_REGRESSION]:
                metrics = ['rmse']
            elif self.task in [consts.TASK_BINARY_CLASSIFICATION, consts.TASK_MULTICLASS_CLASSIFICATION]:
                metrics = ['accuracy']

        y_pred = self.model.predict(X_transformed)
        scores = calc_score(y, y_pred, metrics=metrics, task=self.task,
            pos_label=self.pos_label, classes=self.classes_)

        return scores

    def save(self, model_file):
        with fs.open(f'{model_file}', 'wb') as output:
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(model_file):
        with fs.open(f'{model_file}', 'rb') as input:
            model = pickle.load(input)
            return model


class HyperTS(HyperModel):

    def __init__(self,
                 searcher,
                 dispatcher=None,
                 callbacks=None,
                 reward_metric='accuracy',
                 task=None,
                 discriminator=None,
                 data_cleaner_params=None,
                 cache_dir=None,
                 clear_cache=None):

        self.data_cleaner_params = data_cleaner_params

        HyperModel.__init__(self,
                            searcher,
                            dispatcher=dispatcher,
                            callbacks=callbacks,
                            reward_metric=reward_metric,
                            task=task,
                            discriminator=discriminator)

    def _get_estimator(self, space_sample):
        estimator = HyperTSEstimator(task=self.task, space_sample=space_sample, data_cleaner_params=self.data_cleaner_params)
        return estimator

    def load_estimator(self, model_file):
        return HyperTSEstimator.load(model_file)

    def _get_reward(self, value, key=None):
        def cast_float(value):
            try:
                fv = float(value)
                return fv
            except TypeError:
                return None

        if isinstance(value, dict) and isinstance(key, str):
            reward = cast_float(value[key])
        elif isinstance(value, dict) and not isinstance(key, str):
            reward = cast_float(value[key.__name__])
        else:
            raise ValueError(f'"{key}" should be a string or function name for metric.')

        return reward

    def export_trial_configuration(self, trial):
        return '`export_trial_configuration` does not implemented'

    def search(self, X, y, X_eval, y_eval, cv=False, num_folds=3, max_trials=3, dataset_id=None, trial_store=None, **fit_kwargs):
        ##task

        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)

        if self.searcher.use_meta_learner:
            self.searcher.set_meta_learner(MetaLearner(self.history, dataset_id, trial_store))

        self._before_search()

        # dispatcher = self.dispatcher if self.dispatcher else get_dispatcher(self)
        dispatcher = InProcessDispatcher('/models')   # TODO：

        for callback in self.callbacks:
            callback.on_search_start(self, X, y, X_eval, y_eval,
                                     cv, num_folds, max_trials, dataset_id, trial_store,
                                     **fit_kwargs)
        try:
            trial_no = dispatcher.dispatch(self, X, y, X_eval, y_eval,
                                           cv, num_folds, max_trials, dataset_id, trial_store,
                                           **fit_kwargs)

            for callback in self.callbacks:
                callback.on_search_end(self)
        except Exception as e:
            for callback in self.callbacks:
                callback.on_search_error(self)
            raise e

        self._after_search(trial_no)