import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from hyperts.datasets import load_basic_motions
from hyperts.utils import consts, metrics
from hyperts.experiment import make_experiment
from hyperts.toolbox import random_train_test_split

class Test_Multivariate_Binaryclass_Experiment():

    def test_metrics_acc(self):
        multivariate_binaryclass('accuracy')

    def test_metrics_sk_acc(self):
        multivariate_binaryclass(metrics.accuracy_score)

    def test_metrics_f1(self):
        multivariate_binaryclass('f1')

    def test_metrics_sk_f1(self):
        multivariate_binaryclass(metrics.f1_score)

    def test_metrics_auc(self):
        multivariate_binaryclass('auc')

    def test_metrics_sk_auc(self):
        multivariate_binaryclass(metrics.roc_auc_score)

    def test_metrics_precision(self):
        multivariate_binaryclass('precision')

    def test_metrics_sk_precision(self):
        multivariate_binaryclass(metrics.precision_score)

    def test_metrics_recall(self):
        multivariate_binaryclass('recall')

    def test_metrics_sk_recall(self):
        multivariate_binaryclass(metrics.recall_score)


class Test_Multivariate_Multiclass_Experiment():

    def test_metrics_acc(self):
        multivariate_multiclass('accuracy')

    def test_metrics_sk_acc(self):
        multivariate_multiclass(metrics.accuracy_score)

    def test_metrics_f1(self):
        multivariate_multiclass('f1')

    def test_metrics_sk_f1(self):
        multivariate_multiclass(metrics.f1_score)

    def test_metrics_auc(self):
        multivariate_multiclass('auc')

    def test_metrics_sk_auc(self):
        multivariate_multiclass(metrics.roc_auc_score)

    def test_metrics_precision(self):
        multivariate_multiclass('precision')

    def test_metrics_sk_precision(self):
        multivariate_multiclass(metrics.precision_score)

    def test_metrics_recall(self):
        multivariate_multiclass('recall')

    def test_metrics_sk_recall(self):
        multivariate_multiclass(metrics.recall_score)


def multivariate_binaryclass(metric):
    df = load_basic_motions()
    df['target'] = df['target'].map(lambda x: x if x == 'standing' else 'notstanding')
    train_df, test_df = random_train_test_split(df, test_size=0.2)

    target = 'target'
    task = consts.Task_CLASSIFICATION
    reward_metric = metric
    optimize_direction = consts.OptimizeDirection_MAXIMIZE

    exp = make_experiment(train_df.copy(),
                          mode='dl',
                          task=task,
                          target=target,
                          pos_label='notstanding',
                          reward_metric=reward_metric,
                          optimize_direction=optimize_direction)

    model = exp.run(max_trials=1)

    X_test, y_test = model.split_X_y(test_df.copy())

    y_pred = model.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    assert score >= 0
    print('multivariate_classification accuracy:  {} %'.format(score*100))


def multivariate_multiclass(metric):
    df = load_basic_motions()
    train_df, test_df = random_train_test_split(df, test_size=0.2)

    target = 'target'
    task = consts.Task_CLASSIFICATION
    reward_metric = metric
    optimize_direction = consts.OptimizeDirection_MAXIMIZE

    exp = make_experiment(train_df.copy(),
                          mode='dl',
                          task=task,
                          target=target,
                          pos_label='standing',
                          reward_metric=reward_metric,
                          optimize_direction=optimize_direction)

    model = exp.run(max_trials=1)

    X_test, y_test = model.split_X_y(test_df.copy())

    y_pred = model.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    assert score >= 0
    print('multivariate_classification accuracy:  {} %'.format(score*100))