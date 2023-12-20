from frozenLake import FrozenLake
import control


def find_policy(big_lake=False, gamma=0.9, algorithm='value_iteration'):
    seed = 0

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
        model = control.DynamicProgramming(env)
        policy, value = model.policy_iteration(gamma, theta=0.001, max_iterations=128)
        env.render(policy, value)
    elif algorithm == 'value_iteration':
        print('## Value iteration')
        model = control.DynamicProgramming(env)
        policy, value = model.value_iteration(gamma, theta=0.001, max_iterations=128)
        env.render(policy, value)
    else:
        print('Work In Progress')

if __name__ == '__main__':
    find_policy()

