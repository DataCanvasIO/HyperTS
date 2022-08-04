import os
import copy
import pickle
from sklearn.pipeline import Pipeline

def save_model(model, model_file):
    """Save TSPipeline.

    Parameters
    ----------
    model_file: str, the path where the file is saved. For example, "home/xxx/xxx/models".
    """
    model_file = os.path.join(model_file, f'{model.mode}_models')

    if os.path.exists(model_file):
        raise ValueError(f'The model already exists. Please modify file name {model.mode}_models '
                         f'or delete: {model_file}.')

    os.makedirs(model_file, exist_ok=True)

    if isinstance(model.sk_pipeline, Pipeline) \
            and hasattr(model.sk_pipeline.steps[-1][1], 'save') \
            and hasattr(model.sk_pipeline.steps[-1][1], 'load'):
        submodel = copy.copy(model)
        submodel.sk_pipeline.steps[-1][1].save(model_file, external=True)
        with open(os.path.join(model_file, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(submodel, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(model_file, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model_file):
    """Load TSPipeline.

    Parameters
    ----------
    model_file: str, the path where the file is saved. For example,
        1. "home/xxx/xxx/models/stats_models"
        2. "home/xxx/xxx/models/dl_models"
        3. "home/xxx/xxx/models/nas_models"
    """
    if not os.path.exists(model_file):
        raise ValueError(f'FileNotFoundError: No such file or directory: {model_file}')

    if os.path.isfile(os.path.join(model_file, 'pipeline.pkl')):
        with open(os.path.join(model_file, 'pipeline.pkl'), 'rb') as f:
            pipeline = pickle.load(f)
            assert isinstance(pipeline.sk_pipeline, Pipeline)

        estimator = pipeline.sk_pipeline.steps[-1][1]
        if hasattr(estimator, 'ensemble_size'):
            estimators = estimator.load(f'{model_file}', external=True)
        else:
            estimators = estimator._load(f'{model_file}', mode=pipeline.mode, external=True)
        steps = pipeline.sk_pipeline.steps[:-1] + [(pipeline.sk_pipeline.steps[-1][0], estimators)]
        sk_pipeline = Pipeline(steps)
        pipeline.sk_pipeline = sk_pipeline
    else:
        with open(f'{model_file}/pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)

    return pipeline