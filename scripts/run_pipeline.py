from omegaconf import DictConfig, OmegaConf
import hydra
from pipeline import Pipeline
from utils.utils import set_memory_growth

@hydra.main(version_base=None, config_path='config', config_name='config')
def run_pipeline(cfg: DictConfig):
    Pipeline(cfg).run()

if __name__=='__main__':
    set_memory_growth()
    run_pipeline()