from frozenLake import FrozenLake
import algorithms as rl
import sys
import numpy as np
import config as conf


def main():
    seed = 0

    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9

    print('# Model-based algorithms')

    print('')

    print('## Policy iteration')
    model = rl.TabularModelBased(env, gamma, theta=0.001, max_iterations=128)
    model.policy_iteration()
    env.render(model.policy, model.value)

    print('')

    print('## Value iteration')
    model = rl.TabularModelBased(env, gamma, theta=0.001, max_iterations=128)
    model.value_iteration()
    env.render(model.policy, model.value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 4000

    print('')

    print('## Sarsa')
    model = rl.SARSA(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                     max_iterations=max_episodes, seed=seed)
    model.make_policy()
    env.render(model.policy, model.value)

    print('')

    print('## Q-learning')
    model = rl.Qlearning(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                     max_iterations=max_episodes, seed=seed)
    model.make_policy()
    env.render(model.policy, model.value)

    print('')
    linear_env = rl.LinearWrapper(env)

    print('## Linear Sarsa')

    model = rl.SARSA(linear_env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                     max_iterations=max_episodes, seed=seed)
    model.make_linear_approx_policy()

    linear_env.render(model.policy, model.value)

    print('')

    print('## Linear Q-learning')

    model = rl.Qlearning(linear_env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                         max_iterations=max_episodes, seed=seed)
    model.make_linear_approx_policy()
    linear_env.render(model.policy, model.value)

    print('')

    image_env = rl.FrozenLakeImageWrapper(env)

    print('## Deep Q-network learning')

    model = rl.DeepQLearning(image_env, max_episodes, learning_rate=0.001, gamma=gamma,
                             epsilon=0.2)
    dqn = model.deep_q_network_learning(batch_size=32, target_update_frequency=4, buffer_size=256,
                                        kernel_size=3, conv_out_channels=4,
                                        fc_out_features=8, seed=4)

    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)


if __name__ == '__main__':
    verbose = False # set to True for tracing the algorithm

    conf.init(verbose)

    main()


