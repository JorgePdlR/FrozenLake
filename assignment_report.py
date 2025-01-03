import sys
from datetime import datetime as dt
from matplotlib import pyplot as plt

from frozenLake import FrozenLake
import algorithms as rl
import numpy as np
import config as conf


def plot_returns(model: rl.SARSA | rl.Qlearning | rl.DeepQLearning, gamma: float) -> None:
    """
    Given rewards for each step in the episode, the method calculates
    the sum of discounted rewards for that episode

    Calculates the moving averages of rewards in a given window

    Using matplotlib to plot a line chart to show rewards

    :param model: One of the model free algorithms
    :param algorithm: Name of the algorithm
    :param linear_approx: If the model was found using linear approximation of SARSA or Q Learning
    :param gamma: Discount factor for finding the policy

    :return:
    """
    # Finding sum of discounted rewards for each episode
    episode_discounted_rewards = []
    for rew in model.episode_rewards:
        discounted_sum = 0
        for t, reward in enumerate(rew):
            discount = gamma**t
            discounted_sum += discount * reward
        episode_discounted_rewards.append(discounted_sum)

    # Calculating moving averages of the discounted rewards for each episode
    averages = []
    size = 20
    for i in range(size, len(episode_discounted_rewards)):
        mean = sum(episode_discounted_rewards[i - size:i]) / size
        averages.append(mean)

    # plotting moving averages
    title = model.name +' Return'
    plt.plot(range(len(averages)), averages, linewidth=1, c='m')
    plt.title(title)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Sum of Discounted Rewards', fontsize=12)
    plt.show()


def parameter_search(gamma=0.9) -> None:
    """
    Finds the best parameter values for an algorithm in a given environment

    :param big_lake: Indicates if big lake or small lake will be used as
                     environmental model
    :param gamma: Discount factor
    :param algorithm: Name of the algorithm to execute
    :param linear_approx: Indicates if linear approximation should be used
                           with the provided algorithm (if exists)

    :return:
    """
    print('Running parameter search')
    values = np.arange(0.1, 1, 0.1)

    # Try all possible combinations of values for learning rate and exploration factor
    for algorithm in ['sarsa','q_learning']:
        for lr in values:
            for e in values:
                for big_lake in [False, True]:
                    print('\nUsing Big lake?',big_lake,"\nLearning rate:", lr, "\texploration factor:", e)
                    model = find_policy(big_lake, gamma, algorithm, linear_approx=False, learning_rate=lr, epsilon=e,
                                        stop_at_convergence=True)
                    if model.is_optimal:
                        print('\t\tFound optimal policy in',len(model.episode_rewards),'episodes.')
                    else:
                        print('\t\tDid NOT find optimal policy.')
                    print('*'*50,'\n')

def find_policy(big_lake=False, gamma=0.9, algorithm='value_iteration', linear_approx=False, learning_rate=0.5, epsilon=0.5, stop_at_convergence=False):
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

    :return: Model
    """
    print('*' * 20, algorithm.replace('_',' ').title(), '*' * 20)
    begin = dt.now()
    seed = 0
    theta = 0.001
    max_episodes = 4000
    policy = []

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
            print('With Linear Approximation')
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
            print('With Linear Approximation')
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
        raise Exception('*'*10,algorithm+' Not Implemeted','*'*10)

    completed = dt.now()
    print('Time taken to find policy:',round((completed-begin).total_seconds(),3),'seconds')
    print('Render')
    env.render(model.policy, model.value)

    return model


if __name__ == '__main__':
    big_lake = False
    gamma = 0.9
    algorithm = 'q_learning'
    linear_approx = True  # option only used for sarsa and Q learning
    verbose = False  # set to True for tracing the algorithm
    stop_at_convergence = False

    algorithms = ['policy_iteration', 'value_iteration', 'sarsa', 'q_learning', 'deep_Q_network']
    try:
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
                    gamma = float(i)
                except ValueError:
                    print('Could not parse Gamma value. Using default gamma='+str(gamma))

        conf.init(verbose)
        conf.vprint("Running with verbose", verbose, "big_lake", big_lake, "algorithm", algorithm, "linear_approx",
                    linear_approx, "gamma", gamma)

        parameter_search(gamma)

        # model = find_policy(big_lake, gamma, algorithm, linear_approx, .5, .5, stop_at_convergence)
        # if algorithm in ['sarsa', 'q_learning', 'deep_Q_network']:
        #     plot_returns(model, gamma)

    except Exception as e:
        print('Error while finding policy:\n',e)





