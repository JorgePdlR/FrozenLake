import sys
import numpy as np
import torch as torch
from collections import deque
import frozenLake
import config as conf

def moving_average(rewards, n=20):
    mov_avg = [] #TODO
    for reward_i in rewards[n:]:
        pass

    return mov_avg

def param_search(params):
    pass

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
        self.episode_discounted_rewards = [] # TODO
        self.optimal_policies = []

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
                best_actions = [a for a in range(self.env.n_actions) if np.allclose(qmax, Q[state][a])]
                action = self.random_state.choice(best_actions)
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
                    best_actions = [a for a in range(self.env.n_actions) if np.allclose(qmax, Q[new_state][a])]
                    new_action = self.random_state.choice(best_actions)
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
                best_actions = [a for a in range(self.env.n_actions) if np.allclose(qmax, q_pred[a])]
                action = self.random_state.choice(best_actions)

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
                    best_actions = [a for a in range(self.env.n_actions) if np.allclose(qmax, q_pred_new[a])]
                    new_action = self.random_state.choice(best_actions)

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

class Qlearning:
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
        self.episode_discounted_rewards = [] # TODO
        self.optimal_policies = []

    def make_policy(self):
        # Initiate the q values
        q = np.zeros((self.env.n_states, self.env.n_actions))

        # For all the iterations
        for i in range(self.N):

            s = self.env.reset()  # initial state

            while s != self.env.absorbing_state:

                e = self.random_state.random()
                if e < self.epsilon[i]:
                    a = np.random.choice(range(self.env.n_actions))
                else:
                    qmax = max(q[s])
                    best_actions = [act for act in range(self.env.n_actions) if np.allclose(qmax, q[s][act])]
                    a = self.random_state.choice(best_actions)
                    #a = np.argmax(q[s, :])

                s_prime, r, done = self.env.step(a)

                # update the q value
                q[s, a] += self.alpha[i] * (r + self.gamma * np.max(q[s_prime, :]) - q[s, a])

                # Move to the next state
                s = s_prime

        self.policy = q.argmax(axis=1)
        self.value = q.max(axis=1)
        return self.policy, self.value


    def make_linear_approx_policy(self):
        # Initiate theta
        theta = np.zeros(self.env.n_features)

        eta = self.alpha * (1.0 - np.linspace(0, 1, self.N))
        epsilon = self.epsilon * (1.0 - np.linspace(0, 1, self.N))

        # For all the iterations
        for i in range(self.N):

            # Initial state and feature
            done = False
            s = 0
            f = self.env.reset()

            # Update Q
            q = f.dot(theta)

            # While s is not in an absorbing state
            while s != (self.env.n_states - 1):

                e = self.random_state.random()
                if e < self.epsilon[i]:
                    a = np.random.choice(range(self.env.n_actions))
                else:
                    qmax = max(q)
                    best_actions = [act for act in range(self.env.n_actions) if np.allclose(qmax, q[act])]
                    a = self.random_state.choice(best_actions)
                    #a = np.argmax(q)

                # Take step with action a
                features_prime, r, done = self.env.step(a)

                # Extract s_prime from the features prime
                indices = np.argmax(features_prime, axis=1)
                states = [np.unravel_index(index, (self.env.n_states, self.env.n_actions))[0] for index in
                          indices]
                s_prime = states[0]

                # update delta
                delta = r - q[a]

                # update the value for q
                q = features_prime.dot(theta)

                # Get the best new action with updated q
                # temporal difference
                new_action = np.argmax(q)
                delta += self.gamma * q[new_action]

                # Update the theta value
                theta += eta[i] * delta * f[a, :]
                f = features_prime
                s = s_prime

        self.policy, self.value = self.env.decode_policy(theta)
        return self.policy, self.value


# TODO: class QLearning:
class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env

        lake = self.env.lake

        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])

        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]

        self.state_image = {env.absorbing_state:
                                np.stack([np.zeros(lake.shape)] + lake_image)}
        for state in range(lake.size):
            # Create image of lake shape size
            state_matrix = np.zeros(lake.shape)
            # Get position x,y in current state
            state_y, state_x = np.unravel_index(state, lake.shape)
            # Assign a 1 to the corresponding actual state
            state_matrix[state_y, state_x] = 1
            # Add current state to state_image
            self.state_image.update({state: np.stack([state_matrix] + lake_image)})

    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


class DeepQNetwork(torch.nn.Module):
    def __init__(self, env, learning_rate, kernel_size, conv_out_channels,
                 fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.conv_layer = torch.nn.Conv2d(in_channels=env.state_shape[0],
                                          out_channels=conv_out_channels,
                                          kernel_size=kernel_size, stride=1)

        h = env.state_shape[1] - kernel_size + 1
        w = env.state_shape[2] - kernel_size + 1

        self.fc_layer = torch.nn.Linear(in_features=h * w * conv_out_channels,
                                        out_features=fc_out_features)
        self.output_layer = torch.nn.Linear(in_features=fc_out_features,
                                            out_features=env.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)

        # Convolutional layer
        x = self.conv_layer(x)
        # Rectified linear unit activation
        x = torch.relu(x)
        # Flattening convolutional layer output
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc_layer(x)
        # Rectified linear unit activation
        x = torch.relu(x)
        # Output layer
        x = self.output_layer(x)
        return x

    def train_step(self, transitions, gamma, tdqn):
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions], dtype=np.float32)
        next_states = np.array([transition[3] for transition in transitions], dtype=np.float32)
        dones = np.array([transition[4] for transition in transitions], dtype=np.float32)

        q = self(states)
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)

        target = torch.tensor(rewards, dtype=torch.float32) + gamma * next_q

        # loss function - mean squared of temporal differences
        tmp_diff_loss = torch.nn.MSELoss()
        loss = tmp_diff_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        # Get indices of batch_size from self.buffer without replacement
        transitions_index = self.random_state.choice(len(self.buffer), batch_size, replace=False)
        # Get transitions that correspond to the previous indices
        transitions = [self.buffer[index] for index in transitions_index]
        # Return list of batch_size transitions drawn without replacement
        # from the replay buffer
        return transitions

# TODO: class DeepQLearning:
def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon,
                            batch_size, target_update_frequency, buffer_size,
                            kernel_size, conv_out_channels, fc_out_features, seed):
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                       fc_out_features, seed=seed)
    tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                        fc_out_features, seed=seed)

    epsilon = np.linspace(epsilon, 0, max_episodes)

    for i in range(max_episodes):
        state = env.reset()

        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                with torch.no_grad():
                    q = dqn(np.array([state]))[0].numpy()

                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)

            next_state, reward, done = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())

    return dqn
