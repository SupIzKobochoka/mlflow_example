import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'

MODELS = {model.__name__: model for model in [LogisticRegression, 
                                              RandomForestClassifier, 
                                              LogisticRegressionCV, 
                                              DecisionTreeClassifier]}

def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    params['model_args']['random_state'] = RANDOM_STATE
    logger.info(f'    Параметры модели: {params}')
    model = MODELS[params['model']](**params['model_args'])

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')

    mlflow.log_params({"model_name": params["model"],
                       "random_state": RANDOM_STATE,
                       **params["model_args"]})

    mlflow.sklearn.log_model(model, "model")

if __name__ == '__main__':
    train()