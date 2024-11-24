# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import gymnasium as gym
import optparse
import sys
import os
import random
import numpy as np
import torch

gym.logger.set_level(40)

if "../" not in sys.path:
    sys.path.append("../")

from lib import plotting
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import Solvers.Available_solvers as avs
from lib.envs.gridworld import GridworldEnv
from lib.envs.cluster import ClusterEnv

import matplotlib
import matplotlib.pyplot as plt


def build_parser():
    parser = optparse.OptionParser(
        description="Run a specified RL algorithm on a specified domain."
    )
    parser.add_option(
        "-s",
        "--solver",
        dest="solver",
        type="string",
        default="dqn",
        help="Solver from " + str(avs.solvers),
    )
    parser.add_option(
        "-d",
        "--domain",
        dest="domain",
        type="string",
        default="Cluster",
        help="Domain from OpenAI Gym",
    )
    parser.add_option(
        "-o",
        "--outfile",
        dest="outfile",
        default="out",
        help="Write results to FILE",
        metavar="FILE",
    )
    parser.add_option(
        "-x",
        "--experiment_dir",
        dest="experiment_dir",
        default="Experiments",
        help="Directory to save Tensorflow summaries in",
        metavar="FILE",
    )
    parser.add_option(
        "-e",
        "--episodes",
        type="int",
        dest="episodes",
        default=1000,
        help="Number of episodes for training",
    )
    parser.add_option(
        "-t",
        "--steps",
        type="int",
        dest="steps",
        default=100,
        help="Maximal number of steps per episode",
    )
    parser.add_option(
        "-l",
        "--layers",
        dest="layers",
        type="string",
        default="[24,24]",
        help='size of hidden layers in a Deep neural net. e.g., "[10,15]" creates a net where the'
        "Input layer is connected to a layer of size 10 that is connected to a layer of size 15"
        " that is connected to the output",
    )
    parser.add_option(
        "-a",
        "--alpha",
        dest="alpha",
        type="float",
        default=0.5,
        help="The learning rate (alpha) for updating state/action values",
    )
    parser.add_option(
        "-r",
        "--seed",
        type="int",
        dest="seed",
        default=random.randint(0, 9999999999),
        help="Seed integer for random stream",
    )
    parser.add_option(
        "-g",
        "--gamma",
        dest="gamma",
        type="float",
        default=1.00,
        help="The discount factor (gamma)",
    )
    parser.add_option(
        "-p",
        "--epsilon",
        dest="epsilon",
        type="float",
        default=0.1,
        help="Initial epsilon for epsilon greedy policies (might decay over time)",
    )
    parser.add_option(
        "-P",
        "--final_epsilon",
        dest="epsilon_end",
        type="float",
        default=0.1,
        help="The final minimum value of epsilon after decaying is done",
    )
    parser.add_option(
        "-c",
        "--decay",
        dest="epsilon_decay",
        type="float",
        default=0.99,
        help="Epsilon decay factor",
    )
    parser.add_option(
        "-m",
        "--replay",
        type="int",
        dest="replay_memory_size",
        default=500000,
        help="Size of the replay memory",
    )
    parser.add_option(
        "-N",
        "--update",
        type="int",
        dest="update_target_estimator_every",
        default=10000,
        help="Copy parameters from the Q estimator to the target estimator every N steps.",
    )
    parser.add_option(
        "-b",
        "--batch_size",
        type="int",
        dest="batch_size",
        default=32,
        help="Size of batches to sample from the replay memory",
    )
    parser.add_option(
        "--no-plots",
        help="Option to disable plots if the solver results any",
        dest="disable_plots",
        default=False,
        action="store_true",
    )
    return parser


def readCommand(argv):
    parser = build_parser()
    (options, args) = parser.parse_args(argv)
    return options


def getEnv(domain, render_mode=""):
    if domain == "Cluster":
        return ClusterEnv(
            num_nodes=4,          # 4 nodes in cluster
            max_jobs=10,          # Queue can hold up to 10 jobs
            max_resources=100     # Each resource (CPU/Memory) maxes at 100
            )
    elif domain == "Gridworld":
        return GridworldEnv()
    else:
        try:
            return gym.make(domain, render_mode=render_mode)
        except:
            assert False, "Domain must be a valid (and installed) Gym environment"


def parse_list(string):
    """Convert string representation of list to actual list of integers"""
    string = string.strip('[]')
    return [int(x) for x in string.split(',') if x]


render = False


def on_press(key):
    from pynput import keyboard
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single char keys
    except:
        k = key.name  # other keys
    if k in ["^"]:
        print(f"Key pressed: {k}")
        global render
        render = True


def main(options):
    resultdir = "Results/"

    resultdir = os.path.abspath(f"./{resultdir}")
    options.experiment_dir = os.path.abspath(f"./{options.experiment_dir}")

    # Create result file if one doesn't exist
    print(os.path.join(resultdir, options.outfile + ".csv"))
    if not os.path.exists(os.path.join(resultdir, options.outfile + ".csv")):
        with open(
            os.path.join(resultdir, options.outfile + ".csv"), "w+"
        ) as result_file:
            result_file.write(AbstractSolver.get_out_header())

    random.seed(options.seed)
    env = getEnv(options.domain)
    eval_env = getEnv(options.domain)  # No render_mode needed for new ClusterEnv

    print(f"\n---------- {options.domain} ----------")
    print(f"Domain state space is {env.observation_space}")
    print(f"Domain action space is {env.action_space}")
    print("-" * (len(options.domain) + 22) + "\n")

    # Create solver_params dictionary from options
    solver_params = {
        "hidden_sizes": parse_list(options.layers),
        "lr": options.alpha,
        "replay_memory_size": options.replay_memory_size,
        "batch_size": options.batch_size,
        "gamma": options.gamma,
        "epsilon": options.epsilon,
        "epsilon_decay": options.epsilon_decay,
        "min_epsilon": options.epsilon_end,
        "update_target_every": options.update_target_estimator_every
    }

    # Create solver instance with the parameters
    solver = avs.get_solver_class(options.solver)(env, eval_env, solver_params)

    # For testing without a solver, we can use random actions:
    stats = plotting.EpisodeStats(episode_lengths=[], episode_rewards=[])
    
    with open(os.path.join(resultdir, options.outfile + ".csv"), "a+") as result_file:
        result_file.write("\n")
        for i_episode in range(options.episodes):
            episode_reward = 0
            steps = 0
            obs = env.reset()
            done = False

            obs = env.reset()

            episode_reward = solver.train_episode()    
            
            # Update statistics
            # stats.episode_rewards.append(solver.statistics[Statistics.Rewards.value])
            stats.episode_rewards.append(episode_reward)
            stats.episode_lengths.append(solver.statistics[Statistics.Steps.value])
            print(
                f"Episode {i_episode+1}: Reward {solver.statistics[Statistics.Rewards.value]}, Episode Reward {episode_reward}, Steps {solver.statistics[Statistics.Steps.value]}"
            )
            
            # # Update statistics
            # stats.episode_rewards.append(episode_reward)
            # stats.episode_lengths.append(steps)
            
            # print(f"Episode {i_episode+1}: Reward {episode_reward}, Steps {steps}")
            
            # global render
            # if render and not options.disable_plots:
            #     env.render()
            #     render = False

    if not options.disable_plots:
        # Basic plotting of rewards over episodes
        plt.figure()
        plt.plot(stats.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
        
    return {"stats": stats}


if __name__ == "__main__":
    options = readCommand(sys.argv)
    main(options)
