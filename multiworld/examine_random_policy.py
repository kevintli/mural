import pickle
import argparse
import gym
import skvideo.io
import numpy as np
from pathlib import Path
from multiworld.envs.mujoco import register_goal_example_envs

register_goal_example_envs()

# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Plots rollouts")
    parser.add_argument("-e", "--env_name", 
                        type=str, 
                        help="environment name",
                        default="SawyerDhandInHandPickup-v3")
    parser.add_argument("-p", "--policy", 
                        type=str, 
                        help="path to policy",
                        default="")
    parser.add_argument("-r", "--render", 
                        type=str, 
                        help="onscreen/offscreen rendering",
                        default="onscreen")
    parser.add_argument('-i', '--include', 
                        type=str, 
                        help='task suite to import')
    parser.add_argument('-n', '--num_episodes', 
                        type=int, 
                        default=1,
                        help='number of episodes')
    parser.add_argument('-t', '--horizon_length', 
                        type=int, 
                        default=50,
                        help='rollout length')
    parser.add_argument('-f', '--filename', 
                        type=str, 
                        default='',
                        help='offline rendering video path')
    return parser.parse_args()
def main():
    # get args
    args = get_args()
    # load env
    # if args.include is not "":
    #     exec("import " + args.include)
    env = gym.make(args.env_name)
    rollout_imgs = []
    for ep in range(args.num_episodes):
        env.reset()
        for _ in range(args.horizon_length):
            obs, reward, done, info = env.step(env.action_space.sample())
            rollout_imgs.append(env.render(width=480, height=480, mode="rgb_array"))
    skvideo.io.vwrite(args.filename, np.asarray(rollout_imgs))
    print(f"Done saving videos to {args.filename}")
if __name__ == "__main__":
    main()
    