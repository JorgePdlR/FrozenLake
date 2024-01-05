import numpy as np
import contextlib
import config as conf

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. /n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        # p can be pre-computed. We decided calculate p with p()
        # function instead of pre-computing it and just retrieving
        # its value.

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # Convert possible actions to directions for clarity of reading the code:
        up, left, down, right = [0, 1, 2, 3]
        # Probability of transitioning from state to next_state given an action
        # (return value)
        pt = 0.0
        # Is the agent in a hole ?
        hole = 0
        # Is the agent in the goal ?
        goal = 0

        # Compute if agent is in a hole or the goal. If current state is
        # the absorbing state then we are not in a hole or the goal
        if state != self.absorbing_state:
            hole = True if self.lake_flat[state] == '#' else False
            goal = True if self.lake_flat[state] == '$' else False

        # 1. If the agent is in the absorbing state the probability of remaining in
        #    the same place is 1. Any action taken in the absorbing state leads
        #    to the absorbing state. Validating this first to avoid dealing with
        #    'out of index' issues with the self.lake_flat array since the
        #    absorbing state is in the self.lake.size + 1 position.
        #    The probability of moving from the absorbing state to any other state
        #    is 0, pt is initialized in 0.0 it is not necessary to make the
        #    assignment.
        if state == self.absorbing_state:
            if next_state == self.absorbing_state:
                pt = 1.0
        # 2. If the agent is in a hole or in the goal the probability of moving to
        #    the absorbing state is 1. If the agent is in any position different
        #    from a hole or the goal, and it wants to move to the absorbing state
        #    the probability is 0
        elif next_state == self.absorbing_state or hole or goal:
            if next_state == self.absorbing_state and (hole or goal):
                pt = 1.0
        # 3. Validate if the agent can move from the current state to the next_state.
        #    Consider that it has 4 possible actions: up, down, left, right;
        #    this means that it can move in just 1 direction per action, either
        #    x or y. The absolute change in the coordinates x, y from state to
        #    next_state can't be greater than 1. A diagonal move is invalid since
        #    x = 1 and y = 1 will add up to 2
        else:
            # Get x, y coordinates of state and next_state
            next_state_y, next_state_x = np.unravel_index(next_state, self.lake.shape)
            state_y, state_x = np.unravel_index(state, self.lake.shape)
            # A negative delta_x indicates that the agent should move to the left,
            # positive delta_x indicates it should move to the right
            delta_x = next_state_x - state_x
            # A negative delta_y indicates that the agent should go down, positive
            # delta_y indicate it should go up
            delta_y = next_state_y - state_y

            # If the movement is valid then compute the probability of making the
            # move given the current state, next_state and action. Probability of
            # making an invalid move is 0,
            if (np.abs(delta_x) + np.abs(delta_y)) <= 1:
                # 4. Given that the transition from state to next_state is valid,
                #    validate that by executing the action the agent can get from
                #    state to next_state either by the selected action or by
                #    slipping.

                # Borders of the lake
                border_y, border_x = self.lake.shape
                border_x -= 1
                border_y -= 1

                # Defining walls as the number of borders each tile collides with.
                # Get walls in x and y
                walls_y = 1 if state_y == border_y or state_y == 0 else 0
                walls_x = 1 if state_x == border_x or state_x == 0 else 0
                # Add the number of walls this state collides with x and y
                walls = walls_y + walls_x

                # Transition is from state to next_state, where state is not equal to
                # next_state. In other words, the agent is trying to move from the
                # current state
                if delta_x or delta_y:
                    # For transitioning from state to the next_state the action provided
                    # is the correct one. The agent will get to the next state unless it slips
                    if ((delta_x > 0 and action == right) or (delta_x < 0 and action == left) or
                            (delta_y > 0 and action == down) or (delta_y < 0 and action == up)):
                        pt = (1 - self.slip) + (self.slip / self.n_actions)
                    # For transitioning from state to the next_state the action provided
                    # is incorrect. The only wat to transition is by slipping.
                    else:
                        pt = (self.slip / self.n_actions)
                # Transition is to the same state: state == next_state. The agent can remain in
                # the same state if it crashes with a wall. The number of walls contribute
                # to how probable is to remain in the same state.
                elif walls:
                    # Convert possible values to border places for clarity of reading the code:
                    left_border, right_border, top_border, down_border = [0, border_x, 0, border_y]

                    # The agent is moving to a wall, it will remain in the same state unless is
                    # slips to a place without a wall.
                    if (action == left and state_x == left_border) or (action == right and state_x == right_border) or \
                            (action == up and state_y == top_border) or (action == down and state_y == down_border):
                        pt = (1 - self.slip) + ((self.slip / self.n_actions) * walls)
                    # Action is not in a direction of a wall, the agent will remain in the same
                    # state just if it slips into a wall
                    else:
                        pt = (self.slip / self.n_actions) * walls
                # Is not necessary to put an else case here, since pt was initialized to 0.0
                # If pt has a value of 0.0 at this point means that the agent wanted to remain
                # in the same state but there are no walls to crash to remain in the same place.
                # So the probability of remaining in the same place is 0 given any action,
                # even if the agent slips it will end up moving.

        # Return the probability of moving from state to next_state given an action while
        # considering the probability of slipping
        return pt

    def r(self, next_state, state, action):
        # Reward received by transitioning from state to next_state
        # given an action
        reward = 0.0

        # The agent receives a reward of 1 upon taking an action in the goal
        if state != self.absorbing_state and self.lake_flat[state] == '$':
            reward = 1.0

        # In any other case there is no reward
        return reward

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

def play(big_lake=False):
    seed=0
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
    state = env.reset()
    actions = ['w', 'a', 's', 'd']
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


if __name__ == '__main__':
    play()