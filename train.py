# import argparse
from utils import parse_arguments, set_seed, configure_model, read_yaml, get_current_kortime
from models import Avatar_Generator_Model
import os
# import sys
import wandb


CONFIG_FILENAME = "/home/user/disk2/kdh/AvatarGAN/configs/config.json"
# PROJECT_WANDB = "avatar_image_generator"  # AvatarGAN

# def train(config_file, run_name, run_notes, gpu):
def train(config_file, args):
    set_seed(32)
    wandb_config = read_yaml('/home/user/disk2/kdh/AvatarGAN/configs/wandb.yaml')
    use_wandb = wandb_config['use_wandb']
    config = configure_model(config_file, use_wandb)
    config['gpu'] = args.gpu

    if use_wandb:
        os.environ['WANDB_API_KEY'] = wandb_config['key']
        wandb.login()
        if args.debug:
            wandb_config['project'] = "debug"
        wandb.init(project=wandb_config['project'], config=config, name=args.timestamp, notes=args.run_notes)
        # wandb.init(project=wandb_config['project'], config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False

    xgan = Avatar_Generator_Model(config, use_wandb)
    xgan.load_weights(config.model_path)
    xgan.train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    args = parse_arguments()
    # use_wandb = args.wandb
    # run_name = args.run_name

    # run_notes = args.run_notes
    # gpu = args.gpu
    # debug = args.debug
    timestamp = get_current_kortime()
    args.timestamp = timestamp

    train(CONFIG_FILENAME, args=args)
    # train(CONFIG_FILENAME, run_name=timestamp, run_notes=run_notes, gpu=gpu)
