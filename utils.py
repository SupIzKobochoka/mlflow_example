from typing import Callable
import logging
import os
import yaml
import warnings
import numpy as np

from pathlib import Path
from typing import Any, Mapping, Literal
from ruamel.yaml import YAML

from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore",
                        category=UndefinedMetricWarning)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)

PARAMS_FILEPATH_PATTERN = '/app/params/{stage_name}.yaml'

def load_params(stage_name: str) -> dict:
    params_filepath = PARAMS_FILEPATH_PATTERN.format(stage_name=stage_name)
    if not os.path.exists(params_filepath):
        raise FileNotFoundError(f'Параметров для шага {stage_name} не существует! Проверьте имя шага')
    with open(params_filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params['params']


def get_logger(logger_name: str | None = None,
               level: int = 20,
               ) -> logging.Logger:
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(level)
    return logger


def find_best_thresh(y_true: np.ndarray, 
                     y_probas: np.ndarray,
                     scorer: Callable,
                    ) -> tuple[float, float]:
    '''-> (best_score, best_trash)''' 
    # Буду через сетку перебирать, хоть и не очень правильно
    grid_n = 1_000
    grid = np.arange(0, grid_n) / grid_n
    bool_preds = y_probas[None,:] >= grid[:,None]
    scores = [scorer(y_true, bool_preds[i]) for i in range(grid_n)]
    best_thresh = np.argmax(scores)
    
    return scores[best_thresh], best_thresh / grid_n


def get_scores(scorers_with_thresh: list[callable],
               scorers_without_thresh: list[callable],
               y_true: np.ndarray,
               y_probas: np.ndarray,
               ) -> dict[str, dict[float, float|None]]:
    '''
    -> {scorer_name: {best_score: float,
                      best_thresh, float|None}}
    '''
    scores = {scorer.__name__: {"best_score": scorer(y_true, y_probas),
                                'best_thresh': None
                                } 
              for scorer in scorers_without_thresh}
    for scorer in scorers_with_thresh:
        score, thresh = find_best_thresh(y_true, y_probas, scorer)
        scores[scorer.__name__]  = {'best_score': score, 'best_thresh': thresh}
    
    return scores

def update_yaml_params(stage_name: Literal['evaluate', 'process_data', 'train'], 
                       updates: Mapping[str, Any]
                       ) -> None:
    """
    Частично обновляет params в YAML, 
    комментарии и форматирование сохраняются.
    """

    filepath = Path(PARAMS_FILEPATH_PATTERN.format(stage_name=stage_name))

    if not filepath.exists():
        raise FileNotFoundError(f'Параметров для шага {stage_name} не существует!')

    yaml = YAML()
    yaml.preserve_quotes = True

    with filepath.open("r", encoding="utf-8") as f:
        data = yaml.load(f)

    if "params" not in data or data["params"] is None:
        data["params"] = {}

    for key, value in updates.items():
        data["params"][key] = value

    with filepath.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
