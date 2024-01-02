import sys

import numpy as np
import frozenLake
import config as conf



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
        self.policy = [0]*self.env.n_states
        self.value = [0]*self.env.n_states

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
            conf.vprint("Iteration ", i)
            # Get value of current policy
            value = self.policy_evaluation(policy, gamma, theta, max_iterations)
            # Improve policy
            policy = self.policy_improvement(value, gamma)

            # If policy cannot improve more we have the optimal policy
            equal = prv_policy == policy
            if not np.any(equal == False):
                break

        # Set the optimal policy and value
        self.value = value
        self.policy = policy

    def value_iteration(self, gamma, theta, max_iterations, value=None):
        if value is None:
            value = np.zeros(self.env.n_states)
        else:
            value = np.array(value, dtype=np.float32)

        # Keep iterating while we have budget, or if the delta error is reached
        for i in range(max_iterations):
            conf.vprint("Iteration ", i)
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

        # Set optimal policy and value
        self.value = value
        self.policy = policy


class LinearWrapper:
    def __init__(self, env:frozenLake.FrozenLake):
        self.env = env
        self.lake_flat = self.env.lake_flat
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

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


class SARSA:
    """
    This class implements the SARSA (state-action-reward-state-action) algorithm based
    on Temporal Difference Learning.
    """
    def __init__(self, env: frozenLake.FrozenLake | LinearWrapper, max_iterations=128, learning_rate=0.5, epsilon=0.5,
                 discount_rate=0.9, seed=0):
        self.env = env
        self.N = max_iterations
        self.alpha = np.linspace(learning_rate, 0, max_iterations)
        self.epsilon = np.linspace(epsilon, 0, max_iterations)
        self.gamma = discount_rate
        self.policy = [0]*self.env.n_states
        self.value = [0]*self.env.n_states
        self.random_state = np.random.RandomState(seed)

    def make_policy(self):
        Q = np.zeros((self.env.n_states,self.env.n_actions))

        for i in range(self.N):
            conf.vprint('\nEpisode:',i+1)
            state = self.env.reset()

            # selecting an action based on epsilon greedy policy
            e = self.random_state.random()
            if e < self.epsilon[i]:
                action = self.random_state.choice(self.env.n_actions)
            else:
                qmax = max(Q[state])
                best = [a for a in range(self.env.n_actions) if np.allclose(qmax, Q[state][a])]
                action = self.random_state.choice(best)
            done = False

            while not done:
                conf.vprint('\t\tSteps taken:',self.env.n_steps)
                new_state, reward, done = self.env.step(action)

                # Select a_prime using epsilon greedy approach
                e = self.random_state.random()
                conf.vprint('\t\te:', e, '\tepsilon:', self.epsilon[i])

                # choosing the next action
                if e < self.epsilon[i]:
                    new_action = self.random_state.choice(self.env.n_actions)
                    conf.vprint('\t\tRandom new action chosen')
                else:
                    qmax = max(Q[new_state])
                    best = [a for a in range(self.env.n_actions) if np.allclose(qmax, Q[new_state][a])]
                    new_action = self.random_state.choice(best)
                    conf.vprint('\t\tBEST new action chosen')
                conf.vprint('\t\tState:', state, '\tAction:', action, '\tReward:', reward, '\tNew state:', new_state, '\tNew action:', new_action)
                # temporal difference learning
                temporal_diff = reward + self.gamma*Q[new_state][new_action] - Q[state][action]
                conf.vprint('\t\tTemporal Difference:',temporal_diff)
                Q[state][action] += self.alpha[i]*temporal_diff

                state, action = new_state, new_action
                conf.vprint('\tDone?', done)

        self.policy = np.argmax(Q, axis=1)
        self.value = np.max(Q, axis=1)

    def make_linear_approx_policy(self):
        weights = np.zeros(self.env.n_features)
        conf.vprint('weights',weights.shape)
        for i in range(self.N):
            conf.vprint('\nEpisode:', i+1)
            done = False
            state = self.env.reset()
            conf.vprint('Initial state', state.shape)

            # selecting an action based on epsilon greedy policy
            e = self.random_state.random()
            conf.vprint('\te:',e,'\tepsilon:',self.epsilon[i])

            features = state
            conf.vprint('features:', features.shape)
            q_pred = features.dot(weights)
            conf.vprint('q_pred shape', q_pred)

            if e < self.epsilon[i]:
                action = self.random_state.choice(self.env.n_actions)
            else:
                qmax = max(q_pred)
                best = [a for a in range(self.env.n_actions) if np.allclose(qmax, q_pred[a])]
                action = self.random_state.choice(best)

            while not done:
                conf.vprint('\t\tSteps taken:',self.env.env.n_steps)
                new_state, reward, done = self.env.step(action)
                new_features = new_state
                q_pred_new = new_features.dot(weights)

                conf.vprint('new q',q_pred_new)
                e = self.random_state.random()
                conf.vprint('\t\te:',e,'\tepsilon:',self.epsilon[i])
                if e < self.epsilon[i]:
                    new_action = self.random_state.choice(self.env.n_actions)
                    conf.vprint('\t\tRandom new action chosen',new_action)
                else:
                    qmax = max(q_pred_new)
                    best = [a for a in range(self.env.n_actions) if np.allclose(qmax, q_pred_new[a])]
                    new_action = self.random_state.choice(best)

                    conf.vprint('\t\tBest new action chosen',new_action)

                conf.vprint('\t\tTemporal Difference')
                delta = reward - q_pred[action] + self.gamma*q_pred_new[new_action]
                q_pred = q_pred_new
                conf.vprint('delta',delta)
                conf.vprint('weights', weights.shape)
                conf.vprint('features', features.shape)
                weights += self.alpha[i]*delta*features[action]
                conf.vprint('feature action', features[action])

                state, action = new_state, new_action
                features = state
                conf.vprint('\tDone?', done)
                conf.vprint('state', state.shape)

        self.policy, self.value = self.env.decode_policy(weights)
