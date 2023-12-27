import numpy as np
import frozenLake


class TabularModelBased:
    """
    This class implements Tabular model-based algorithms:
        - Policy evaluation
        - Policy improvement
        - Policy iteration
        - Value iteration
    """
    def __init__(self, env):
        self.env = env

    def policy_evaluation(self, policy, gamma, theta, max_iterations):
        value = np.zeros(self.env.n_states, dtype=np.float32)

        # Loop the number of iterations or while our error is greater than
        # the tolerance parameter
        for i in range(max_iterations):
            delta = 0
            # Go through all tha states
            for state in range(self.env.n_states):
                # Store value of the state previous to any modification
                v = value[state]
                policy_state_product = 0

                # We have just one action per state in a deterministic policy, is not necessary
                # to loop through all the actions. Leaving this 'for' just to match the policy
                # evaluation formula
                # Go through all the actions in the policy
                for a in range(self.env.n_actions):
                    policy_p = 0
                    # If the action is not part of the policy don't calculate anything
                    if a == policy[state]:
                        policy_p = 1
                    else:
                        continue

                    state_product = 0

                    # Go through all the possible next states
                    for state_prime in range(self.env.n_states):
                        # Get probability of transitioning from state to state_prime
                        # given action
                        pt = self.env.p(state_prime, state, a)
                        # Get reward of transitioning from state to state_prime given
                        # action
                        rt = self.env.r(state_prime, state, a)

                        state_product += pt * (rt + (gamma * value[state_prime]))

                    # Store product from all possible action in policy
                    policy_state_product += state_product * policy_p

                # Update value of this state
                value[state] = policy_state_product

                # Calculate the difference between our previous value and the new one
                delta = max(delta, abs(v - value[state]))

            # If our delta is smaller than the error value theta, stop
            if delta < theta:
                break

        return value

    def policy_improvement(self, value, gamma):
        improved_policy = np.zeros(self.env.n_states, dtype=int)

        # Go through all the states
        for state in range(self.env.n_states):
            q = list()

            # Go through all the actions
            for a in range(self.env.n_actions):
                state_prime_q = 0

                # Go through all next states from the actual state
                for state_prime in range(self.env.n_states):
                    # Get probability of transitioning from state to state_prime
                    # given action
                    pt = self.env.p(state_prime, state, a)
                    # Get reward of transitioning from state to state_prime given
                    # action
                    rt = self.env.r(state_prime, state, a)

                    state_prime_q += pt * (rt + (gamma * value[state_prime]))

                # Store value of all actions of possible next states in a list
                q.append(state_prime_q)

            q = np.array(q)

            # Get the best action in the current state
            improved_policy[state] = np.argmax(q)

        # Return improved policy
        return improved_policy

    def policy_iteration(self, gamma, theta, max_iterations, policy=None):
        if policy is None:
            policy = np.zeros(self.env.n_states, dtype=int)
        else:
            policy = np.array(policy, dtype=int)

        value = np.zeros(self.env.n_states, dtype=np.float32)

        # Iterate the maximum number of iterations or while there are still
        # improvements in the policy
        for i in range(max_iterations):
            prv_policy = policy
            # print("Iteration ", i)
            # Get value of current policy
            value = self.policy_evaluation(policy, gamma, theta, max_iterations)
            # Improve policy
            policy = self.policy_improvement(value, gamma)

            # If policy cannot improve more we have the optimal policy
            equal = prv_policy == policy
            if not np.any(equal == False):
                break

        # Return the optimal policy and value of tha policy
        return policy, value

    def value_iteration(self, gamma, theta, max_iterations, value=None):
        if value is None:
            value = np.zeros(self.env.n_states)
        else:
            value = np.array(value, dtype=np.float32)

        # Keep iterating while we have budget, or if the delta error is reached
        for i in range(max_iterations):
            # print("Iteration ", i)
            delta = 0

            # Go through all the states
            for state in range(self.env.n_states):
                v = value[state]
                policy_state_product = list()

                # Go through all the actions
                for a in range(self.env.n_actions):
                    state_product = 0

                    # Go through all the next states given the current state
                    for state_prime in range(self.env.n_states):
                        pt = self.env.p(state_prime, state, a)
                        rt = self.env.r(state_prime, state, a)

                        state_product += pt * (rt + (gamma * value[state_prime]))

                    # Store evaluation of states in a list, we will decide which is
                    # the best outside this loop
                    policy_state_product.append(state_product)

                # Update value whit best evaluation
                policy_state_product = np.array(policy_state_product)
                value[state] = np.max(policy_state_product)

                # Update delta
                delta = max(delta, abs(v - value[state]))

            # If our delta is smaller than the error value theta, stop
            if delta < theta:
                break

        # Update policy given the optimal value
        policy = self.policy_improvement(value, gamma)

        # Return optimal policy and value
        return policy, value

class SARSA:
    """
    This class implements the SARSA (state-action-reward-state-action) algorithm based
    on Temporal Difference Learning.
    """
    def __init__(self, env:frozenLake.FrozenLake, max_iterations, learning_rate=0.1, epsilon=0.01, discount_rate=0.9, tabular=True):
        self.env = env
        self.N = max_iterations
        self.alpha = learning_rate
        self.epsilon = np.linspace(epsilon, 0, max_iterations)
        self.gamma = discount_rate
        self.state_init = env.reset()
        self.policy = [0]*env.n_states
        self.value = [0]*self.env.n_states
        # up, left, down, right = [0, 1, 2, 3]

    def make_policy(self):
        Q = [[0]*self.env.n_actions]*self.env.n_states

        for i in self.N:
            state = self.state_init
            # selection an action based on epsilon greedy policy
            e = np.random.random()
            if e<self.epsilon[i]:
                action = np.random.choice(range(self.env.n_actions))
            else:
                action = np.argmax(Q[state])
            done = False

            while not done:
                new_state, reward, done = self.env.step(action)
                # Select a_prime using epsilon greedy approach
                e = np.random.random()
                if e < self.epsilon[i]:
                    new_action = np.random.choice(range(self.env.n_actions))
                else:
                    new_action = np.argmax(Q[state])
                # temporal difference learning
                Q[state][action] += self.alpha*(reward +
                                                self.gamma*Q[new_state][new_action] -
                                                Q[state][action])
                state, action = new_state, new_action

        self.policy[s] = np.argmax(Q[s], axis=1)
        self.value[s] = np.max(Q[s], axis=1)


class Q_Learning:
    """This class implements the Q-learning algorithm."""

    def __init__(self, env: frozenLake.FrozenLake, max_iterations, learning_rate=0.5, epsilon=0.5, discount_rate=0.9):
        self.env = env
        self.N = max_iterations
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = discount_rate
        self.state_init = env.lake[0][0]
        self.policy = [0] * env.n_states
        self.value = [0] * env.n_states
        # up, left, down, right = [0, 1, 2, 3]
        # Expected policy [2 1 2 1 2 0 2 0 3 2 2 0 0 3 3 0 0]

    def make_policy(self, seed=101):

        # Initiate the q values
        q = np.zeros((self.env.n_states, self.env.n_actions))

        # For all the iterations
        for i in range(self.N):

            s = self.env.reset()  # initial state

            while s != self.env.absorbing_state:

                e = np.random.random()
                if e < self.epsilon:
                    a = np.random.choice(range(self.env.n_actions))
                else:
                    a = np.argmax(q[s, :])

                s_prime, r, done = self.env.step(a)

                # update the q value
                q[s, a] += self.alpha * (r + self.gamma * np.max(q[s_prime, :]) - q[s, a])
                print(q[s, a])
                # Move to the next state
                s = s_prime

        self.policy = q.argmax(axis=1)
        self.value = q.max(axis=1)
        return self.policy, self.value


class LinearWrapper:
    def __init__(self, env):
        self.env = env
        self.n_actions = env.n_actions
        self.n_states = env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        print(s)
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        print(features)
        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


class Linear_Q_Learning:
    def __init__(self, env: frozenLake.FrozenLake, max_iterations, learning_rate=0.5, epsilon=0.5, discount_rate=0.9):
        self.env = env
        self.linear_env = LinearWrapper(env)
        self.N = max_iterations
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = discount_rate
        self.state_init = env.lake[0][0]
        self.policy = [0] * env.n_states
        self.value = [0] * env.n_states
        # up, left, down, right = [0, 1, 2, 3]
        # [2 1 2 1 2 0 2 0 3 2 2 0 0 3 3 0 0]

    def make_policy(self, seed=101):

        theta = np.zeros(self.linear_env.n_features)
        q = np.zeros(self.linear_env.n_actions)

        # For all the iterations
        for i in range(self.N):

            s = self.state_init
            f = self.linear_env.reset()

            for a in range(self.env.n_actions):
                q[a] = np.dot(theta,f[a])

            # features = self.linear_env.encode_state(s)
            # q = np.dot(features, theta).flatten()

            while s != self.env.absorbing_state:

                e = np.random.random()
                if e < self.epsilon:
                    a = np.random.choice(range(self.linear_env.n_actions))
                else:
                    a = np.argmax(q)

                features_prime, r, done = self.linear_env.step(a)
                delta = r - q[a]

                for a_prime in range(self.env.n_actions):
                    q[a_prime] = np.dot(theta, f[a_prime])

                #q_prime = np.dot(features_prime, theta)
                delta = delta + self.gamma * np.max(q)
                print(f.shape)
                print(delta)
                theta += self.alpha * (f * delta)
                s = f

        self.policy, self.value = self.linear_env.decode_policy(theta)
        self.linear_env.render(self.policy, self.value)
