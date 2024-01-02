from frozenLake import FrozenLake
import algorithms as rl
import sys
import numpy as np
import config as conf

def find_policy(big_lake=False, gamma=0.9, algorithm='value_iteration',
                linear_approx=False):
    seed = 0
    theta = 0.001
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
        model = rl.TabularModelBased(env)
        model.policy_iteration(gamma, theta=theta, max_iterations=128)
        env.render(model.policy, model.value)
    elif algorithm == 'value_iteration':
        print('*'*10,'Value iteration','*'*10)
        model = rl.TabularModelBased(env)
        model.value_iteration(gamma, theta=theta, max_iterations=128)
        env.render(model.policy, model.value)
    elif algorithm == 'sarsa':
        print('*'*10,'SARSA CONTROL','*'*10)
        if big_lake:
            max_episodes = 5000
        else:
            max_episodes = 4000

        if linear_approx:
            env = rl.LinearWrapper(env)
            model = rl.SARSA(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                                  max_iterations=max_episodes, seed=seed)
            model.make_linear_approx_policy()

        else:
            model = rl.SARSA(env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                                  max_iterations=max_episodes, seed=seed)
            model.make_policy()
        print('Policy:',model.policy)
        print('Value:',model.value)
        env.render(model.policy, model.value)
    elif algorithm == 'deep_Q_network':
        # Recreate frozen lake to update max_steps in case of the big lake
        env = FrozenLake(lake, slip=0.1, max_steps=np.array(lake).size, seed=seed)

        image_env = rl.FrozenLakeImageWrapper(env)
        print('*'*10,'Deep Q-network learning','*'*10)

        dqn = rl.deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                         gamma=gamma, epsilon=0.2, batch_size=32,
                                         target_update_frequency=4, buffer_size=256,
                                         kernel_size=3, conv_out_channels=4,
                                         fc_out_features=8, seed=4)
        policy, value = image_env.decode_policy(dqn)
        image_env.render(policy, value)
    else:
        print('*'*10,'Work In Progress','*'*10)

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
    model = rl.TabularModelBased(env)
    model.policy_iteration(gamma, theta=0.001, max_iterations=128)
    env.render(model.policy, model.value)

    print('')

    print('## Value iteration')
    model = rl.TabularModelBased(env)
    model.value_iteration(gamma, theta=0.001, max_iterations=128)
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
    #policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
    #                           epsilon=0.5, seed=seed)
    #env.render(policy, value)

    print('')
    linear_env = rl.LinearWrapper(env)

    print('## Linear Sarsa')

    model = rl.SARSA(linear_env, learning_rate=0.5, discount_rate=gamma, epsilon=0.5,
                     max_iterations=max_episodes, seed=seed)
    model.make_linear_approx_policy()

    linear_env.render(model.policy, model.value)

    print('')

    print('## Linear Q-learning')

    #parameters = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma,
    #                               epsilon=0.5, seed=seed)
    #policy, value = linear_env.decode_policy(parameters)
    #linear_env.render(policy, value)

    print('')

    image_env = rl.FrozenLakeImageWrapper(env)

    print('## Deep Q-network learning')

    dqn = rl.deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                     gamma=gamma, epsilon=0.2, batch_size=32,
                                     target_update_frequency=4, buffer_size=256,
                                     kernel_size=3, conv_out_channels=4,
                                     fc_out_features=8, seed=4)

    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)


if __name__ == '__main__':
    big_lake = False
    gamma = 0.9
    algorithm = 'sarsa'
    linear_approx = False
    verbose = False

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

    find_policy(big_lake, gamma, algorithm, linear_approx)
    #main()


