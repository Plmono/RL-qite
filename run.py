""" Run script """
import argparse
import os
import gym_cut
import torch

from common.util import learn, parse_all_args


""" Some notice """
print("""
    Notes:
        CUDA usage is depend on `CUDA_VISIBLE_DEVICES`;
        Log will be recorded at ../logs/{env}_{algorithm}_{seed}/ by default;
        If you need multi-gpu training or other nn specific features, please 
        modify the default.py file in corresponding algorithm folder.
""")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


""" Parse arguments """
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--env', type=str,default=' ', help='environment ID')
parser.add_argument('--algorithm', type=str,default='ppo',  help='Algorithm')
parser.add_argument('--nenv', type=int, default=12, help='parrallel number')
parser.add_argument('--env_type', type=str,default='classic_control',
                    )
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--number_timesteps', type=float, default=9e6)
parser.add_argument('--reward_scale', type=float, default=1.0)
parser.add_argument('--save_path', type=str, default='../checkpoints')
parser.add_argument('--save_interval', type=int, default=0,
                    help='save model every x steps (0 = disabled)')
parser.add_argument('--log_path', type=str, default='../logs',
                    help='save model every x steps (0 = disabled)')
common_options, other_options = parse_all_args(parser)


""" Learn """
if __name__ == '__main__':
    if common_options.save_interval:
        os.makedirs(common_options.save_path, exist_ok=True)
    model = learn(
        device=device,
        env_id='maxcut-v1',
        nenv=common_options.nenv,
        env_type=common_options.env_type,
        seed=common_options.seed,
        number_timesteps=int(common_options.number_timesteps),
        save_path=common_options.save_path,
        save_interval=common_options.save_interval,
        log_path=common_options.log_path,
        algorithm=common_options.algorithm,
        reward_scale=common_options.reward_scale,
        **other_options
    )
