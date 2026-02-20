import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
from utils import get_logger, load_params

PREPROCESSORS = {preprocessor.__name__: preprocessor for preprocessor in [OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler]}
STAGE_NAME = 'process_data'

def process_data():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали скачивать данные')
    dataset = load_dataset(DATASET_NAME)
    logger.info('Успешно скачали данные!')

    logger.info('Делаем предобработку данных')
    df = dataset['train'].to_pandas()
    target_column = 'income'
    columns = df.columns if params['features'] == 'all' else params['features']
    drop_cols = set(params['cols_to_drop'])
    columns = [col for col in columns if col not in drop_cols and col != target_column]

    X, y = df[columns], df[target_column]
    logger.info(f'    Используемые фичи: {columns}')

    all_cat_features = params['cats']
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    y: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=TEST_SIZE, 
                                                        shuffle=True, 
                                                        stratify=y, 
                                                        random_state=RANDOM_STATE
                                                        )

    X_train, X_test, y_train, y_test = map(lambda x: x.reset_index(drop=True), [X_train, X_test, y_train, y_test])

    if isinstance(params['train_size'], float):
        assert params['train_size'] <= 1, f'{params['train_size']=} must be <= 1'
        train_len = X_train.shape[0]
        X_train = X_train.iloc[np.arange(0, int(train_len * params['train_size']))] # Можем, потому что shuffle=True
        y_train = y_train.iloc[np.arange(0, int(train_len * params['train_size']))]

    elif isinstance(params['train_size'], int):
        assert X_train.shape[0] >= params['train_size'], f"{params['train_size']=} must be <=  {X_train.shape[0]=}"
        X_train = X_train.iloc[np.arange(0, params['train_size'])] # Можем, потому что shuffle=True
        y_train = y_train.iloc[np.arange(0, params['train_size'])]

    cats_preprocessor = PREPROCESSORS[params['cats_encoder']['name']](**params['cats_encoder']['params'])

    if params['num_scalers']['name'] is not None:
        num_preprocessor = PREPROCESSORS[params['num_scalers']['name']](**params['num_scalers']['params'])
    else:
        num_preprocessor = Pipeline([("_", "passthrough")])

    X_train_cats = cats_preprocessor.fit_transform(X_train[cat_features])
    X_test_cats = cats_preprocessor.transform(X_test[cat_features])

    X_train_num = num_preprocessor.fit_transform(X_train[num_features])
    X_test_num = num_preprocessor.transform(X_test[num_features])

    X_train = np.hstack([X_train_cats, X_train_num])
    X_test = np.hstack([X_test_cats, X_test_num])

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    logger.info('Начали сохранять датасеты')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)
    for split, split_name in zip((X_train, X_test, y_train, y_test),
                                 ('X_train', 'X_test', 'y_train', 'y_test'),
                                 ):
        pd.DataFrame(split).to_csv(DATASET_PATH_PATTERN.format(split_name=split_name), index=False)
    logger.info('Успешно сохранили датасеты!')

    mlflow.log_params({"dataset_name": DATASET_NAME,
                    "n_features": len(columns),
                    "features": ",".join(columns),
                    "train_size_rows": int(len(y_train)),
                    "test_size_rows": int(len(y_test)),
                    "cat_features": ",".join(cat_features),
                    "num_features": ",".join(num_features),
                    "cats_encoder_name": params['cats_encoder']['name'],
                    "cats_encoder_params": str(params['cats_encoder']['params']),
                    "num_scaler_name": params['num_scalers']['name'],
                    "num_scaler_params": str(params['num_scalers']['params']),
                    "random_state": RANDOM_STATE})
    dataset_dir = os.path.dirname(DATASET_PATH_PATTERN)
    mlflow.log_artifacts(dataset_dir, artifact_path="dataset")

if __name__ == '__main__':
    process_data()