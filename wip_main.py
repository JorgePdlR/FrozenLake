from frozenLake import FrozenLake
import algorithms as rl
import sys
import numpy as np
import config as conf


def parameter_search(big_lake, gamma, algorithm, linear_approx):
    iterations = 9
    lr = np.linspace(.1, .9, iterations)
    ep = np.linspace(.1, .9, iterations)

    for i in lr:
        for j in ep:
            print("Using learning rate", i, "exploration factor", j)
            find_policy(big_lake, gamma, algorithm, linear_approx, i, j)


def find_policy(big_lake=False, gamma=0.9, algorithm='value_iteration',
                linear_approx=False, learning_rate=0.5, epsilon=0.5):
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
        print('*'*10,'Policy iteration','*'*10)
        model = rl.TabularModelBased(env, gamma, theta=0.001, max_iterations=128)
        model.policy_iteration()
        env.render(model.policy, model.value)
    elif algorithm == 'value_iteration':
        print('*'*10,'Value iteration','*'*10)
        model = rl.TabularModelBased(env)
        model.value_iteration()
        env.render(model.policy, model.value)
    elif algorithm == 'sarsa':
        print('*'*10,'SARSA CONTROL','*'*10)

        if linear_approx:
            env = rl.LinearWrapper(env)
            model = rl.SARSA(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                             max_iterations=max_episodes, seed=seed, stop_optimal=False)
            model.make_linear_approx_policy()

        else:
            model = rl.SARSA(env, learning_rate=learning_rate, discount_rate=gamma, epsilon=epsilon,
                             max_iterations=max_episodes, seed=seed, stop_optimal=False)
            model.make_policy()
        print('Policy:',model.policy)
        print('Value:',model.value)
        env.render(model.policy, model.value)
        rl.moving_average(model.episode_discounted_rewards)
    elif algorithm == 'q':
        print('*' * 10, 'Q learning Work In Progress', '*' * 10)
        if linear_approx:
            env = rl.LinearWrapper(env)
            model = rl.Qlearning(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                                 max_iterations=max_episodes, seed=seed, stop_optimal=False)
            model.make_linear_approx_policy()

        else:
            model = rl.Qlearning(env, learning_rate=learning_rate, discount_rate=gamma, epsilon=epsilon,
                                 max_iterations=max_episodes, seed=seed, stop_optimal=False)
            model.make_policy()
        env.render(model.policy, model.value)
        # rl.moving_average(model.episode_discounted_rewards)
    elif algorithm == 'deep_Q_network':
        # Recreate frozen lake to update max_steps in case of the big lake
        env = FrozenLake(lake, slip=0.1, max_steps=np.array(lake).size, seed=seed)

        image_env = rl.FrozenLakeImageWrapper(env)
        print('*'*10,'Deep Q-network learning','*'*10)

        model = rl.DeepQLearning(image_env, max_episodes, learning_rate=0.001, gamma=gamma,
                                 epsilon=0.2)
        dqn = model.deep_q_network_learning(batch_size=32, target_update_frequency=4, buffer_size=256,
                                            kernel_size=3, conv_out_channels=4,
                                            fc_out_features=8, seed=4)

        policy, value = image_env.decode_policy(dqn)
        image_env.render(policy, value)
    else:
        print('*'*10,'Work In Progress','*'*10)

if __name__ == '__main__':
    big_lake = False
    gamma = 0.9
    algorithm = 'sarsa'
    linear_approx = False
    verbose = False  # set to True for tracing the algorithm

    for i in sys.argv:
        if i == "-v" or i == "-verbose" or i == "v" or i == "verbose":
            verbose = True
        elif i == "big_lake" or i == "-big_lake":
            big_lake = True
        elif i == "linear_approx" or i == "-linear_approx":
            linear_approx = True
        elif (i == "sarsa" or i == "policy_iteration" or i == "value_iteration" or i == "value_iteration"
              or i == "deep_Q_network"):
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

    find_policy(big_lake, gamma, algorithm, linear_approx, .5, .5)
    #parameter_search(big_lake, gamma, algorithm, linear_approx)




