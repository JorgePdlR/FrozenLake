import sys
from datetime import datetime as dt
from matplotlib import pyplot as plt

from frozenLake import FrozenLake
import algorithms as rl
import numpy as np
import config as conf


def sum_discounted_reward(episode_rewards:[], gamma:float) -> float:
    """
    Given rewards for each step in the episode, the method calculates
    the sum of discounted rewards for that episode
    :param episode_rewards: Rewards accumulated at each step in the episode
    :param gamma: discount factor
    :return:
    """
    discounted_sum = 0
    for t in (range(len(episode_rewards))):
        pgamma = gamma**t
        discounted_sum += pgamma * episode_rewards[t]

    return discounted_sum


def moving_average(rewards:[], size=20) -> []:
    """
    Calculates the moving averages of rewards in a given window
    :param rewards: Rewards per episode
    :param size: Window size
    :return: `size` window moving averages
    """
    sample_avgs = []
    for i in range(size, len(rewards)):
        mean = sum(rewards[i - size:i])/size
        sample_avgs.append(mean)

    return sample_avgs


def plot_rewards(y:[], algorithm:str) -> None:
    """
    Using matplotlib to plot a line chart to show rewards
    :param y: Rewards
    :param algorithm: for the title
    :return:
    """
    plt.plot(range(len(y)), y, linewidth=1, c='m')
    plt.title(algorithm+' Return')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Sum of Discounted Rewards', fontsize=12)
    plt.show()


def parameter_search(big_lake:bool, gamma:float, algorithm:str, linear_approx:bool):
    """
    Finds the best parameter values for an algorithm in a given environment
    :param big_lake: Indicates if big lake or small lake will be used as
                     environmental model
    :param gamma: Discount factor
    :param algorithm: Name of the algorithm to execute
    :param linear_approx: Indicates if linear approximation should be used
                           with the provided algorithm (if exists)
    :return: Nothing
    """
    values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Try all possible combinations of values with itself. It represents
    # learning rate and exploration factor
    for i in values:
        for j in values:
            print("Using learning rate", i, "exploration factor", j)
            find_policy(big_lake, gamma, algorithm, linear_approx, i, j)


def find_policy(big_lake=False, gamma=0.9, algorithm='value_iteration',
                linear_approx=False, learning_rate=0.5, epsilon=0.5, stop_at_convergence=False):
    """
    Find a policy for the indicated environment using the model provided in algorithm
    :param big_lake: to use big_lake environment or not
    :param gamma: discount factor
    :param algorithm: model based or model-free algorithms
    :param linear_approx: to use linear approximation function with model-free algorithm
                            (only used for sarsa & Q learning)
    :param learning_rate: step size to improve policy
    :param epsilon: used to calculate epsilon for E-greedy policy methods
    :param stop_at_convergence: stop running the algorithm for more episodes if it converges
                            (it has no impact on tabular model based algorithm)
    :return:
    """
    print('*' * 10, algorithm.replace('_',' ').title(), '*' * 10)
    begin = dt.now()
    seed = 0
    theta = 0.001
    max_episodes = 4000
    policy = 0

    if big_lake:
        lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]
        max_steps = 64
    else:
        # Small lake
        lake = [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        max_steps = 16

    env = FrozenLake(lake, slip=0.1, max_steps=max_steps, seed=seed)

    conf.vprint("Running with gamma", gamma, "max_steps", max_steps, "theta", theta)
    if algorithm == 'policy_iteration':
        model = rl.TabularModelBased(env, gamma, theta=0.001, max_iterations=128)
        model.policy_iteration()

    elif algorithm == 'value_iteration':
        model = rl.TabularModelBased(env)
        model.value_iteration()

    elif algorithm == 'sarsa':
        if linear_approx:
            env = rl.LinearWrapper(env)
            model = rl.SARSA(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                             max_iterations=max_episodes, seed=seed, stop_optimal=stop_at_convergence)
            model.make_linear_approx_policy()

        else:
            model = rl.SARSA(env, learning_rate=learning_rate, discount_rate=gamma, epsilon=epsilon,
                             max_iterations=max_episodes, seed=seed, stop_optimal=stop_at_convergence)
            model.make_policy()

    elif algorithm == 'q_learning':
        if linear_approx:
            env = rl.LinearWrapper(env)
            model = rl.Qlearning(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                                 max_iterations=max_episodes, seed=seed, stop_optimal=False)
            model.make_linear_approx_policy()

        else:
            model = rl.Qlearning(env, learning_rate=learning_rate, discount_rate=gamma, epsilon=epsilon,
                                 max_iterations=max_episodes, seed=seed, stop_optimal=False)
            model.make_policy()

    elif algorithm == 'deep_Q_network':
        model = rl.DeepQLearning(env, max_episodes, learning_rate=0.001, gamma=gamma,
                                 epsilon=0.2, seed=4)
        model.make_policy(batch_size=32, target_update_frequency=4, buffer_size=256,
                          kernel_size=3, conv_out_channels=4, fc_out_features=8)

        model.env.render(model.policy, model.value)

    else:
        print('*'*10,algorithm+' Not Implemeted','*'*10)
        sys.exit(1)
    completed = dt.now()
    print('Time taken to find policy:',round((completed-begin).total_seconds(),3),'seconds')

    print('Render')
    env.render(model.policy, model.value)

    # Moving averages of sum of discounted rewards for model-free methods
    if algorithm in ['sarsa', 'q_learning', 'deep_Q_network']:
        episode_discounted_rewards = []
        for episode in model.episode_rewards:
            discounted_rewards = sum_discounted_reward(episode, gamma)
            episode_discounted_rewards.append(discounted_rewards)
        averages = moving_average(episode_discounted_rewards)
        # plotting moving averages
        title = algorithm.replace('_', ' ').title()
        if linear_approx:
            title += ' (Linear Approximation)'
        plot_rewards(averages, title)


if __name__ == '__main__':
    big_lake = False
    gamma = 0.9
    algorithm = 'q_learning'
    linear_approx = True # option only used for sarsa and Q learning
    verbose = False  # set to True for tracing the algorithm
    stop_at_convergence = False

    algorithms = ['policy_iteration', 'value_iteration', 'sarsa', 'q_learning', 'deep_Q_network']
    for i in sys.argv:
        if i == "-v" or i == "-verbose" or i == "v" or i == "verbose":
            verbose = True
        elif i == "big_lake" or i == "-big_lake":
            big_lake = True
        elif i == "linear_approx" or i == "-linear_approx":
            linear_approx = True
        elif i in algorithms:
            algorithm = str(i)
        else:
            try:
                float(i)
                gamma = float(i)
            except ValueError:
                None

    conf.init(verbose)
    conf.vprint("Running with verbose", verbose, "big_lake", big_lake, "algorithm", algorithm, "linear_approx",
                linear_approx, "gamma", gamma)

    find_policy(big_lake, gamma, algorithm, linear_approx, .5, .5, stop_at_convergence)

    # # parameter_search(big_lake, gamma, algorithm, linear_approx)





