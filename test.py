"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(opt.world, opt.stage, actions,
                           "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))
    model = PPO(env.observation_space.shape[0], len(actions))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["flag_get"]:
            print("World {} stage {} completed".format(opt.world, opt.stage))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
