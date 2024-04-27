
from parameter import *
from trainer import Trainer
from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
import torch

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
                             config.batch_size, config.num_workers,shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)

    if config.train:
        if config.model=='sagan':
            trainer = Trainer(data_loader.loader(), config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    else:
        # python main.py --train False --pretrained_model 996975 --batch_size 64 --imsize 64 --version sagan_celeb
        tester = Tester(config)
        tester.test()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    config = get_parameters()
    print(config)
    main(config)
