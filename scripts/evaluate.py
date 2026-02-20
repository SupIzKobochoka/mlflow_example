import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os
import pandas as pd
from joblib import load
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             average_precision_score)
from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params, get_scores
import mlflow

STAGE_NAME = 'evaluate'

SCORERS = {s.__name__: s for s in [accuracy_score,
                                   precision_score,
                                   recall_score,
                                   f1_score,
                                   roc_auc_score,
                                   average_precision_score]}


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError('Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением')
    model = load(MODEL_FILEPATH)

    logger.info('Скорим модель на тесте')
    y_proba = model.predict_proba(X_test)[:, 1]

    scores = get_scores(scorers_with_thresh=[SCORERS[s] for s in params['metrics_with_threshold']],
                        scorers_without_thresh=[SCORERS[s] for s in params['metrics_without_threshold']],
                        y_true=y_test,
                        y_probas=y_proba
                        )
    logger.info(f'Значения скоров: \n{scores}')

    # PR curve кривая
    logger.info('Строим и логируем PR-кривую в MLflow')
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curve')
    pr_curve_path = 'pr_curve.png'
    fig.savefig(pr_curve_path, dpi=150, bbox_inches='tight')
    logger.info('PR-кривая залогирована: plots/pr_curve.png')
    plt.close(fig)

    mlflow.log_artifact(pr_curve_path, artifact_path='plots')
    mlflow.log_metrics({metric:scores[metric]['best_score'] for metric in scores})

if __name__ == '__main__':
    evaluate()