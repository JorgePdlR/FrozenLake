from frozenLake import FrozenLake
import control
import LinearWrapper
import deepQnetwork as deepQ
import sys
import numpy as np


def find_policy(big_lake=True, gamma=0.9, algorithm='value_iteration'):
    seed = 0
    max_episodes = 4000

    if big_lake:
        lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]
    else:
        # Small lake
        lake = [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    if algorithm == 'policy_iteration':
        print('## Policy iteration')
        model = control.TabularModelBased(env)
        policy, value = model.policy_iteration(gamma, theta=0.001, max_iterations=128)
        env.render(policy, value)
    elif algorithm == 'value_iteration':
        print('## Value iteration')
        model = control.TabularModelBased(env)
        policy, value = model.value_iteration(gamma, theta=0.001, max_iterations=128)
        env.render(policy, value)

    elif algorithm == 'Q_Learning':
        policy, value = control.Q_Learning(env, 10000).make_policy()
        env.render(policy, value)

    elif algorithm == 'Linear_Q_Learning':
        linear_env = LinearWrapper.LinearWrapper(env)
        policy, value = control.Linear_Q_Learning(linear_env, 8000).make_policy()
        linear_env.render(policy, value)

    elif algorithm == 'deep_Q_network':
        # Recreate frozen lake to update max_steps in case of the big lake
        env = FrozenLake(lake, slip=0.1, max_steps=np.array(lake).size, seed=seed)

        image_env = deepQ.FrozenLakeImageWrapper(env)
        print('## Deep Q-network learning')

        dqn = deepQ.deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                            gamma=gamma, epsilon=0.2, batch_size=32,
                                            target_update_frequency=4, buffer_size=256,
                                            kernel_size=3, conv_out_channels=4,
                                            fc_out_features=8, seed=4)
        policy, value = image_env.decode_policy(dqn)
        image_env.render(policy, value)
    else:
        print('Work In Progress')


if __name__ == '__main__':
    big_lake = False
    gamma = 0.9
    algorithm = 'Linear_Q_Learning'

    if len(sys.argv) == 4:
        big_lake = bool(sys.argv[1])
        gamma = float(sys.argv[2])
        algorithm = str(sys.argv[3])
    else:
        print('Parameters not provided, using default')

    find_policy(big_lake, gamma, algorithm)

