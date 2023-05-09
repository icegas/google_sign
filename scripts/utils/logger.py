from mlflow import log_metric, log_param, log_artifacts, log_artifact
import mlflow
import subprocess
import shutil
from pathlib import Path

class Logger():
    def __init__(self, cfg) -> None:
        mlflow.set_tracking_uri(cfg.tracking_uri) 
        mlflow.set_experiment(cfg.experiment_name)
        mlflow.set_tag("mlflow.runName", cfg.run_name)
        log_artifacts("config", "config")
    
    def end_run(self):
        mlflow.end_run()

    def log_epoch(self, metrics, model_path):

        for metric_name, metric in metrics.items():
            log_metric(metric_name, metric)
        
        #shutil.make_archive(model['path'][:-4], 'zip', model['path'][:-3] + 'tf')
        log_artifact(model_path, Path(model_path).stem + '.tf')
        
    

