import sys
import numpy as np
import torch as torch
from collections import deque
import frozenLake
import config as conf


def get_optimal(env, gamma):
    model = TabularModelBased(env, gamma, theta=0.001, max_iterations=128)
    model.policy_iteration()

    return model

class TabularModelBased:
    """
    This class implements Tabular model-based algorithms:
        - Policy evaluation
        - Policy improvement
        - Policy iteration
        - Value iteration
    """
    def __init__(self, env, gamma=0.9, theta=0.001, max_iterations=128):
        self.env = env
        self.policy = [0]*self.env.n_states
        self.value = [0]*self.env.n_states
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

    def policy_evaluation(self, policy) -> []:
        """
        Given a policy returns the expected return of starting in any state
        s and following the policy
        :param policy:
        :return: V(s) for all the states of the provided policy
        """
        value = np.zeros(self.env.n_states, dtype=np.float32)

        # Loop the number of iterations or while our error is greater than
        # the tolerance parameter
        for i in range(self.max_iterations):
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

                        state_product += pt * (rt + (self.gamma * value[state_prime]))

                    # Store product from all possible action in policy
                    policy_state_product += state_product * policy_p

                # Update value of this state
                value[state] = policy_state_product

                # Calculate the difference between our previous value and the new one
                delta = max(delta, abs(v - value[state]))

            # If our delta is smaller than the error value theta, stop
            if delta < self.theta:
                break

        return value

    def policy_improvement(self, value) -> []:
        """
        Implements policy improvement algorithm. Returns a policy that is
        V(s)' >= V(s) for all s in S. If there is no improvement then provided
        policy is already optimal. (Caller should validate if the returned policy
        is the same as the provided one)
        :param value: V(s) for all the states in the environment
        :return: Improved policy
        """
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

                    state_prime_q += pt * (rt + (self.gamma * value[state_prime]))

                # Store value of all actions of possible next states in a list
                q.append(state_prime_q)

            q = np.array(q)

            # Get the best action in the current state
            improved_policy[state] = np.argmax(q)

        # Return improved policy
        return improved_policy

    def policy_iteration(self, policy=None) -> None:
        """
        Policy iteration algorithm. Applies a sequence of evaluations and improvements
        to an arbitrary (or provided) initial policy. When the policy cannot improve
        it is guaranteed that the policy is optimal by the Bellman optimality equations.
        Once the optimal policy is found policy and V(s) for that policy are stored in
        the class parameters policy and value
        :param policy: policy to improve (can be omitted)
        :return:
        """
        if policy is None:
            policy = np.zeros(self.env.n_states, dtype=int)
        else:
            policy = np.array(policy, dtype=int)

        value = np.zeros(self.env.n_states, dtype=np.float32)

        # Iterate the maximum number of iterations or while there are still
        # improvements in the policy
        for i in range(self.max_iterations):
            prv_policy = policy
            conf.vprint("Iteration ", i)
            # Get value of current policy
            value = self.policy_evaluation(policy)
            # Improve policy
            policy = self.policy_improvement(value)

            # If policy cannot improve more we have the optimal policy
            equal = prv_policy == policy
            if not np.any(equal == False):
                break

        # Set the optimal policy and value
        self.value = value
        self.policy = policy

    def value_iteration(self, value=None) -> None:
        """
        Value iteration algorithm. Improves the estimates for the values of each
        state. Arbitrary or provided V for all the stages converges to V*. Once
        V* is found policy and V(s) are stored in the class parameters policy
        and value
        :param value: V(s) for all the states in the environment (can be ommited)
        :return:
        """
        if value is None:
            value = np.zeros(self.env.n_states)
        else:
            value = np.array(value, dtype=np.float32)

        # Keep iterating while we have budget, or if the delta error is reached
        for i in range(self.max_iterations):
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

                        state_product += pt * (rt + (self.gamma * value[state_prime]))

                    # Store evaluation of states in a list, we will decide which is
                    # the best outside this loop
                    policy_state_product.append(state_product)

                # Update value whit best evaluation
                policy_state_product = np.array(policy_state_product)
                value[state] = np.max(policy_state_product)

                # Update delta
                delta = max(delta, abs(v - value[state]))

            # If our delta is smaller than the error value theta, stop
            if delta < self.theta:
                break

        # Update policy given the optimal value
        policy = self.policy_improvement(value)

        # Set optimal policy and value
        self.value = value
        self.policy = policy


class LinearWrapper:
    """
    Wrapper class for model-free algorithms using linear approximation function.
    Helps to one-hot encode states and decode policy from the features.
    """
    def __init__(self, env:frozenLake.FrozenLake):
        self.env = env
        self.lake_flat = self.env.lake_flat
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s: int) -> []:
        """
        Returns a one-hot action encoded action-state features
        :param s: Position of the agent in the flattened lake
        :return:
        """
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta) -> ([],[]):
        """
        Given a set of action-state features, returns the policy and value
        :param theta: action-state features
        :return: tuple of decoded policy[] & value[]
        """
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self) -> []:
        """
        Resets the environment and sends the agent to the starting point
        :return: Returns features (of the initial state) after resetting the environment
        """
        return self.encode_state(self.env.reset())

    def step(self, action) -> ([], int, bool):
        """
        Given an action to perform in a state, the function returns the next state,
        reward gained by taking that step and if the stoping criteria is reached
        based on one of these conditions:
        - reached max_steps
        - reached an absorbing state (goal or hole)
        :param action:
        :return: tuple of new state features, reward, done (stopping criteria)
        """
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        """
        Shows a visual representation of the policy and value of states
        :param policy: Best action to take
        :param value:
        :return:
        """
        self.env.render(policy, value)


class SARSA:
    """
    This class implements the SARSA (state-action-reward-state-action) algorithm, an On-Policy Temporal Difference Learning method.
    The policy is calculated using a 1 step state-action-reward-new_state-new_action algorithm,
    i.e. take an action in the current state, which lands the agent in a new state and gets a reward for that action.
    Then uses epsilon-greedy policy to take the next action.
    Temporal difference is used to calculate the Q value of that state-action pair.
    """
    def __init__(self, env: frozenLake.FrozenLake | LinearWrapper, max_iterations=128, learning_rate=0.5, epsilon=0.5,
                 discount_rate=0.9, seed=0, stop_optimal=False):
        self.env = env
        self.N = max_iterations
        self.alpha = np.linspace(learning_rate, 0, max_iterations)
        self.epsilon = np.linspace(epsilon, 0, max_iterations)
        self.gamma = discount_rate
        self.policy = [0]*self.env.n_states
        self.value = [0]*self.env.n_states
        self.random_state = np.random.RandomState(seed)
        self.stop_optimal = stop_optimal
        self.episode_rewards = []
        self.is_optimal = False
        self.name = 'SARSA'

    def make_policy(self) -> None:
        """
        Given an environment and algorithm parameters, this method iterates over `max_iterations` number of episodes
        to discover the optimal policy & value (and stores it as class attribute).
        Temporal difference is used to calculate the Q value of that state-action pair.
        :return:
        """
        Q = np.zeros((self.env.n_states,self.env.n_actions))
        if self.stop_optimal:
            optimal_policy = get_optimal(self.env, self.gamma)

        for i in range(self.N):
            conf.vprint('\nEpisode:',i+1)
            state = self.env.reset()
            episode_rewards = []

            # selecting an action based on epsilon greedy policy
            e = self.random_state.random()
            if e < self.epsilon[i]:
                action = self.random_state.choice(self.env.n_actions)
            else:
                qmax = max(Q[state])
                best_actions = [a for a in range(self.env.n_actions) if np.allclose(qmax, Q[state][a])]
                action = self.random_state.choice(best_actions)
            done = False
            # run till max_steps is exhausted or reach an absorbing state
            while not done:
                conf.vprint('\t\tSteps taken:',self.env.n_steps)
                new_state, reward, done = self.env.step(action)
                episode_rewards.append(reward)

                # Select new_action using epsilon greedy approach
                e = self.random_state.random()
                conf.vprint('\t\te:', e, '\tepsilon:', self.epsilon[i])
                if e < self.epsilon[i]:
                    new_action = self.random_state.choice(self.env.n_actions)
                    conf.vprint('\t\tRandom new action chosen')
                else:
                    # if there are multiple best actions then randomly choose from among them
                    qmax = max(Q[new_state])
                    best_actions = [a for a in range(self.env.n_actions) if np.allclose(qmax, Q[new_state][a])]
                    new_action = self.random_state.choice(best_actions)
                    conf.vprint('\t\tBEST new action chosen')
                conf.vprint('\t\tState:', state, '\tAction:', action, '\tReward:', reward, '\tNew state:', new_state, '\tNew action:', new_action)
                # temporal difference learning for 1-step sarsa
                temporal_diff = reward + self.gamma*Q[new_state][new_action] - Q[state][action]
                conf.vprint('\t\tTemporal Difference:',temporal_diff)
                Q[state][action] += self.alpha[i]*temporal_diff

                state, action = new_state, new_action
                conf.vprint('\tDone?', done)
            self.episode_rewards.append(episode_rewards)

            if self.stop_optimal:
                policy = np.argmax(Q, axis=1)
                self_policy_val = optimal_policy.policy_evaluation(policy)
                if (self_policy_val == optimal_policy.value).all():
                    self.is_optimal = True
                    break

        self.policy = np.argmax(Q, axis=1)
        self.value = np.max(Q, axis=1)

    def make_linear_approx_policy(self) -> None:
        """
        Given an environment and algorithm parameters, this method iterates over `max_iterations` number of episodes
        using a linear approximation function by finding weights for each action-feature pair and then finding the
        optimal policy & value (and stores it as class attribute).
        :return:
        """
        self.name += ' with Linear Function Approximation'
        weights = np.zeros(self.env.n_features)
        conf.vprint('weights',weights.shape)
        for i in range(self.N):
            conf.vprint('\nEpisode:', i+1)
            done = False
            state = self.env.reset()
            conf.vprint('Initial state', state.shape)

            episode_rewards = []

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

                episode_rewards.append(reward)

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
    """
    This class implements the Q Learning algorithm, an Off-Policy Temporal Difference Learning method.
    The policy is calculated using a 1 step state-action-reward-new_state algorithm.
    Temporal difference is used to calculate the Q value of that state-action pair.
    """
    def __init__(self, env: frozenLake.FrozenLake | LinearWrapper, max_iterations=128, learning_rate=0.5, epsilon=0.5,
                 discount_rate=0.9, seed=0, stop_optimal=False):
        self.env = env
        self.N = max_iterations
        self.alpha = np.linspace(learning_rate, 0, max_iterations)
        self.epsilon = np.linspace(epsilon, 0, max_iterations)
        self.gamma = discount_rate
        self.policy = [0]*self.env.n_states
        self.value = [0]*self.env.n_states
        self.random_state = np.random.RandomState(seed)
        self.episode_rewards = []
        self.stop_optimal = stop_optimal
        self.name = 'Q Learning'

    def make_policy(self) -> None:
        """
        Given an environment and algorithm parameters, this method iterates over `max_iterations` number of episodes
        to discover the optimal policy & value (and stores it as class attribute)
        Temporal difference is used to calculate the Q value of that state-action pair.
        :return:
        """
        if self.stop_optimal:
            optimal_policy = get_optimal(self.env, self.gamma)
        # Initiate the q values
        q = np.zeros((self.env.n_states, self.env.n_actions))

        # For all the iterations
        for i in range(self.N):
            done = False
            s = self.env.reset()  # initial state
            episode_rewards = []
            while not done:

                # Choose action based on epsilon-greedy policy
                e = self.random_state.random()
                if e < self.epsilon[i]:
                    a = np.random.choice(range(self.env.n_actions))
                else:
                    qmax = max(q[s])
                    best_actions = [act for act in range(self.env.n_actions) if np.allclose(qmax, q[s][act])]
                    a = self.random_state.choice(best_actions)

                # Take a step
                s_prime, r, done = self.env.step(a)
                episode_rewards.append(r)

                # Update Q value using temporal difference
                q[s, a] += self.alpha[i] * (r + self.gamma * np.max(q[s_prime, :]) - q[s, a])

                # Move to the next state
                s = s_prime

            self.episode_rewards.append(episode_rewards)

            if self.stop_optimal:
                policy = np.argmax(q, axis=1)
                self_policy_val = optimal_policy.policy_evaluation(policy)
                if (self_policy_val == optimal_policy.value).all():
                    self.is_optimal = True
                    break

        self.policy = q.argmax(axis=1)
        self.value = q.max(axis=1)

    def make_linear_approx_policy(self) -> None:
        """
        Given an environment and algorithm parameters, this method iterates over `max_iterations` number of episodes
        using a linear approximation functions by finding weights for each action-feature pair
        to discover the optimal policy & value (and stores it as class attribute)
        :return:
        """
        self.name += ' with Linear Function Approximation'
        if self.stop_optimal:
            optimal_policy = get_optimal(self.env, self.gamma)
        # Initiate theta
        theta = np.zeros(self.env.n_features)

        # For all the iterations
        for i in range(self.N):
            # Initial state and feature
            done = False
            f = self.env.reset()
            episode_rewards = []

            # Update Q
            q = f.dot(theta)
            # While s is not in an absorbing state and not reached max_steps
            while not done:
                # Choose action based on epsilon-greedy policy
                e = self.random_state.random()
                if e < self.epsilon[i]:
                    a = np.random.choice(range(self.env.n_actions))
                else:
                    qmax = max(q)
                    best_actions = [act for act in range(self.env.n_actions) if np.allclose(qmax, q[act])]
                    a = self.random_state.choice(best_actions)

                # Take step with action a
                features_prime, r, done = self.env.step(a)
                episode_rewards.append(r)
                # update delta
                delta = r - q[a]

                # update the value for q
                q = features_prime.dot(theta)

                # Get the best new action with updated q
                # temporal difference
                new_action = np.argmax(q)
                delta += self.gamma * q[new_action]

                # Update the theta value
                theta += self.alpha[i] * delta * f[a, :]
                f = features_prime

            self.episode_rewards.append(episode_rewards)
            if self.stop_optimal:
                current_policy, _ = self.env.decode_policy(theta)
                self_policy_val = optimal_policy.policy_evaluation(current_policy)
                if (self_policy_val == optimal_policy.value).all():
                    self.is_optimal = True
                    break

        self.policy, self.value = self.env.decode_policy(theta)


class FrozenLakeImageWrapper:
    """
    Creates an "image representation" of the frozen lake environment
    compose of 4 channels of size h x w where h is the number of rows
    and w is the number of columns of the lake grid:
    - First channel: All elements are 0 except of the agent position
    - Second channel: All elements are 0 except the start tile
    - Third channel: All elements are 0 except the hole tiles
    - Forth channel: All elements are 0 except the goal tile
    """
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
        """
        Returns the corresponding images given the state
        :param state: environment state
        :return: Image of the state
        """
        return self.state_image[state]

    def decode_policy(self, dqn):
        """
        Decodes the policy and value for each state of the provided
        Deep Q network
        :param dqn: Deep Q network
        :return: Policy and values for each state
        """
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        """
        Returns default state encoded in an image
        :param:
        :return: Default state encoded in an image
        """
        return self.encode_state(self.env.reset())

    def step(self, action):
        """
        Executes action in the current state
        :param action: action to execute in the current state
        :return: New state, reward, True if
                 terminal state otherwise False
        """
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None) -> None:
        """
        Renders policy, value and environment
        :param policy: Policy to render
        :param value: Value for each state
        :return:
        """
        self.env.render(policy, value)


class DeepQNetwork(torch.nn.Module):
    """
    Convolutional neural network used by deep Q-network learning algorithm
    """
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
        """
        Computes forward pass of the model
        :param x: input values
        :return: forward result
        """
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

    def train_step(self, transitions, gamma, tdqn) -> None:
        """
        Step in the training process of the deep Q-network
        :param transitions: Transitions from replay buffer
        :param gamma: Discount factor
        :param tdqn: Target Q-network
        :return:
        """
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
    """
    Stores the agent's experience (s,a,r,s+1) or transition
    at each time step in a buffer.
    """
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition) -> None:
        """
        Add new transition to the buffer
        :param transition: Transition to store
        :return:
        """
        self.buffer.append(transition)

    def draw(self, batch_size):
        """
        Return a batch of the stored transitions in the buffer.
        Transitions in the batch are uniformly drawn from the buffer
        :param batch_size: Size of the batch.
        :return: Batch of transitions
        """
        # Get indices of batch_size from self.buffer without replacement
        transitions_index = self.random_state.choice(len(self.buffer), batch_size, replace=False)
        # Get transitions that correspond to the previous indices
        transitions = [self.buffer[index] for index in transitions_index]
        # Return list of batch_size transitions drawn without replacement
        # from the replay buffer
        return transitions


class DeepQLearning:
    """
    Deep Q-network learning algorithm. Learns successful policies directly from
    high-dimensional sensory inputs. Adaptation from:
    [Mnih et al., 2015] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A.,
     Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland,
     A. K., Ostrovski, G., et al. (2015). Human-level control through deep
     reinforcement learning. Nature, 518(7540):529.
    """
    def __init__(self, env, max_episodes=128, learning_rate=0.5, epsilon=0.5,
                 gamma=0.9, seed=0):
        self.env = FrozenLakeImageWrapper(env)
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.seed = seed
        self.episode_rewards = []
        self.policy = [0]*env.n_states
        self.value = [0]*env.n_states
        self.name = 'Deep Q Learning'

    def make_policy(self, batch_size, target_update_frequency, buffer_size,
                    kernel_size, conv_out_channels, fc_out_features) -> None:
        """
        Trains deep Q-network to learn a policy from the provided image
        encoded environment
        :param batch_size: Number of transitions in the batch
        :param target_update_frequency: How often target Q network is updated
        :param buffer_size: Size of the replay buffer
        :param kernel_size: Kernel size for CNN
        :param conv_out_channels: Number of CNN outputs
        :param fc_out_features: Number of output features
        :return:
        """
        random_state = np.random.RandomState(self.seed)
        replay_buffer = ReplayBuffer(buffer_size, random_state)

        dqn = DeepQNetwork(self.env, self.learning_rate, kernel_size, conv_out_channels,
                           fc_out_features, seed=self.seed)
        tdqn = DeepQNetwork(self.env, self.learning_rate, kernel_size, conv_out_channels,
                            fc_out_features, seed=self.seed)

        epsilon = np.linspace(self.epsilon, 0, self.max_episodes)

        for i in range(self.max_episodes):
            state = self.env.reset()

            episode_rewards = []

            done = False
            while not done:
                if random_state.rand() < epsilon[i]:
                    action = random_state.choice(self.env.n_actions)
                else:
                    with torch.no_grad():
                        q = dqn(np.array([state]))[0].numpy()

                    qmax = max(q)
                    best = [a for a in range(self.env.n_actions) if np.allclose(qmax, q[a])]
                    action = random_state.choice(best)

                next_state, reward, done = self.env.step(action)

                episode_rewards.append(reward)

                replay_buffer.append((state, action, reward, next_state, done))

                state = next_state

                if len(replay_buffer) >= batch_size:
                    transitions = replay_buffer.draw(batch_size)
                    dqn.train_step(transitions, self.gamma, tdqn)

            if (i % target_update_frequency) == 0:
                tdqn.load_state_dict(dqn.state_dict())

            self.episode_rewards.append(episode_rewards)

        self.policy, self.value = self.env.decode_policy(dqn)
