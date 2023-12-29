from frozenLake import FrozenLake
import control
import deepQnetwork as deepQ
import sys
import numpy as np


def find_policy(big_lake=False, gamma=0.9, algorithm='value_iteration',
                linear_approx=False):
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

    env = FrozenLake(lake, slip=0.1, max_steps=256, seed=seed)

    if algorithm == 'policy_iteration':
        print('*'*10,'Policy iteration','*'*10)
        model = control.TabularModelBased(env)
        policy, value = model.policy_iteration(gamma, theta=0.001, max_iterations=128)
        env.render(policy, value)
    elif algorithm == 'value_iteration':
        print('*'*10,'Value iteration','*'*10)
        model = control.TabularModelBased(env)
        policy, value = model.value_iteration(gamma, theta=0.001, max_iterations=128)
        env.render(policy, value)
    elif algorithm == 'sarsa':
        print('*'*10,'SARSA CONTROL','*'*10)
        if big_lake:
            max_episodes = 5000
        else:
            max_episodes = 1000

        if linear_approx:
            env = control.LinearWrapper(env)
            model = control.SARSA(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.9,
                                  max_iterations=max_episodes, seed=seed)
            model.make_linear_approx_policy()

        else:
            model = control.SARSA(env, learning_rate=0.5, discount_rate=gamma, epsilon=1,
                                  max_iterations=max_episodes, seed=seed)
            model.make_policy()
        print('Policy:',model.policy)
        print('Value:',model.value)
        env.render(model.policy, model.value)
    elif algorithm == 'deep_Q_network':
        # Recreate frozen lake to update max_steps in case of the big lake
        env = FrozenLake(lake, slip=0.1, max_steps=np.array(lake).size, seed=seed)

        image_env = deepQ.FrozenLakeImageWrapper(env)
        print('*'*10,'Deep Q-network learning','*'*10)

        dqn = deepQ.deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                            gamma=gamma, epsilon=0.2, batch_size=32,
                                            target_update_frequency=4, buffer_size=256,
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

    if len(sys.argv) == 4:
        big_lake = bool(sys.argv[1])
        gamma = float(sys.argv[2])
        algorithm = str(sys.argv[3])
    else:
        print('Parameters not provided, using default\n')

    find_policy(big_lake, gamma, algorithm, linear_approx=True)

