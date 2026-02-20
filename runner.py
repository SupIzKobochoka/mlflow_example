from scripts import evaluate, process_data, train
import mlflow

MLFLOW_URI = "http://158.160.2.37:5000/"
EXPERIMENT_NAME = "homework_klevcov"

def set_mlflow():

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def run_experiment(run_name):
    with mlflow.start_run(run_name=run_name):
        process_data()
        train()
        evaluate()

if __name__ == '__main__':
    set_mlflow()
    run_experiment(run_name="test_run")
