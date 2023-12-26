import numpy as np
import torch as torch
from collections import deque
import frozenLake


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

        # TODO: Computing forward model
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

        # TODO: the loss is the mean squared error between `q` and `target`
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
        # TODO:
        # Get indices of batch_size from self.buffer without replacement
        transitions_index = self.random_state.choice(len(self.buffer), batch_size, replace=False)
        # Get transitions that correspond to the previous indices
        transitions = [self.buffer[index] for index in transitions_index]
        # Return list of batch_size transitions drawn without replacement
        # from the replay buffer
        return transitions


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