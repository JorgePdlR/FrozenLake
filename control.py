import numpy as np


class DynamicProgramming:
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
    def __init__(self):
        pass


class Q:
    def __init__(self):
        pass