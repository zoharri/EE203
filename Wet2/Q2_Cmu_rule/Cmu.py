import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Cmu():
    def __init__(self, probs, costs):
        self.probs = probs
        self.costs = costs
        self.N = len(costs)
        self.n_states = 2 ** self.N

    def _get_binary_state(self, state):
        """
        Converts a state to a binary representation
        :param state: int represented state
        :return: binary representation of state
        """
        b_str = np.binary_repr(state, self.N)
        b_list = [int(c) for c in b_str]
        return np.asarray(b_list)

    def _calc_state_cost(self, state_b):
        """
        This function will implement the total cost function
        :param state_b: binary representation of the state
        :return: the cost of the state
        """
        total_cost = 0
        for i in range(self.N):
            if state_b[i]:
                # this means that the i'th job is still waiting in the queue
                total_cost += self.costs[i]
        return total_cost

    def _bellman_equation_one_step(self, action, curr_val, state):
        """
        Calculates one step of the bellman equation given a current value function, action and state. This can be used
        in the policy evaluation and policy iteration for example.
        :param action: The action for the bellman equation
        :param curr_val: The value function (represented as a vector)
        :param state: The state (represented as int)
        :return: The value function calculated from the bellman equation for this state and policy
        """
        next_state_val_exp = 0
        b_start_state = self._get_binary_state(state)
        job = action
        assert b_start_state[job] != 0, "Invalid policy, you can only choose an unfinished job"
        next_state_1 = b_start_state.copy()
        next_state_2 = b_start_state.copy()
        next_state_2[job] = 0
        next_state_1_int = next_state_1.dot(1 << np.arange(next_state_1.shape[-1] - 1, -1, -1))
        next_state_2_int = next_state_2.dot(1 << np.arange(next_state_2.shape[-1] - 1, -1, -1))

        next_state_val_exp += curr_val[next_state_1_int] * (1 - self.probs[job])
        next_state_val_exp += curr_val[next_state_2_int] * self.probs[job]
        return -self._calc_state_cost(self._get_binary_state(state)) + next_state_val_exp

    def policy_eval(self, policy):
        """
        This function will evaluate the value function for a fixed (and given) policy
        :param policy: a vector of length self.n_states, corresponding to the given policy
        :return: a vector of length self.n_states, corresponding to the value function
        """
        curr_val = np.zeros(self.n_states)
        next_val = np.zeros(self.n_states)
        while True:
            for i in range(1, self.n_states):
                next_val[i] = self._bellman_equation_one_step(policy[i], curr_val, i)
            if np.array_equal(curr_val, next_val):
                return curr_val
            curr_val = next_val.copy()

    def get_cost_policy(self):
        """
        This will calculate the policy with the rule argmax(c) over the left jobs
        :return: The calculated policy
        """
        policy = np.zeros(self.n_states, dtype=np.uint8)
        for i in range(self.n_states):
            b_state = list(self._get_binary_state(i))
            max_c_job = 0
            max_c = 0
            for job in range(self.N):
                if b_state[job] == 1:
                    # we can only choose from the unfinished jobs
                    if self.costs[job] > max_c:
                        max_c = costs[job]
                        max_c_job = job
            policy[i] = max_c_job
        return policy

    def get_cost_mu_policy(self):
        """
        This will calculate the optimal policy with the rule argmax(c*mu) over the left jobs
        :return: The calculated policy
        """
        policy = np.zeros(self.n_states, dtype=np.uint8)
        for i in range(self.n_states):
            b_state = list(self._get_binary_state(i))
            max_c_mu_job = 0
            max_c_mu = 0
            for job in range(self.N):
                if b_state[job] == 1:
                    # we can only choose from the unfinished jobs
                    if self.costs[job] * self.probs[job] > max_c_mu:
                        max_c_mu = self.costs[job] * self.probs[job]
                        max_c_mu_job = job
            policy[i] = max_c_mu_job
        return policy

    def get_rand_policy(self):
        """
        Calculate a random and valid policy
        :return: random policy
        """
        policy = np.zeros(self.n_states, dtype=np.uint8)
        for i in range(1, self.n_states):
            b_i = self._get_binary_state(i)
            pos_actions = np.where(b_i == 1)[0]
            policy[i] = np.random.choice(pos_actions)
        return policy

    def policy_iter(self):
        """
        calculation of the optimal policy using the policy iteration algorithm
        :return: The calculated policy, The value function of the greedy policy during the training at s_0
        """

        s_0_val = []

        curr_policy = self.get_cost_policy()
        next_policy = curr_policy.copy()

        num_iter = 0
        while True:
            num_iter += 1
            curr_val = self.policy_eval(curr_policy)
            s_0_val.append(curr_val[-1])
            for i in range(1, self.n_states):
                best_job = 0
                best_job_val = -np.infty
                for job_i, job_not_finished in enumerate(self._get_binary_state(i)):
                    if job_not_finished:
                        job_val = self._bellman_equation_one_step(job_i, curr_val, i)
                        if best_job_val < job_val:
                            best_job_val = job_val
                            best_job = job_i
                next_policy[i] = best_job

            if np.array_equal(next_policy, curr_policy):
                print("number of iterations of the policy iter: {}".format(num_iter))
                return curr_policy, s_0_val
            curr_policy = next_policy.copy()

    def simulate_trans(self, state, action):
        """
        The simulation of the environment for the learning section =
        :param state: current state (as int)
        :param action: action to take in the given state
        :return: next_state, cost
        """
        b_state = self._get_binary_state(state)
        if b_state[action] == 0:
            return state, 0
        else:
            next_state = b_state.copy()
            if np.random.rand() < self.probs[action]:
                # action successful
                next_state[action] = 0
            next_state = next_state.dot(1 << np.arange(next_state.shape[-1] - 1, -1, -1))
            return next_state, self._calc_state_cost(b_state)

    def plot_value(self, value_function, symbol, color):
        """
        Simple plot of the value function
        :param value_function: the value function to plot
        :param symbol: symbol for the plot
        :param color: color for the plot
        :return: None
        """
        x = list(range(self.n_states))
        plt.plot(x, value_function, symbol, color=color)
        # plt.title("Value function of the 'cost only' policy")
        plt.xlabel("State [decimal representation]")
        plt.ylabel("Value funciton")

    def plot_policy(self, policy, symbol, color):
        """
        Simple plot of the policy
        :param policy: the policy to plot
        :param symbol: symbol for the plot
        :param color: color for the plot
        :return: None
        """
        x = list(range(self.n_states))
        plt.plot(x, policy + 1, symbol, color=color)
        plt.xlabel("State [decimal representation]")
        plt.ylabel("Policy")

    def TD0(self, policy, true_value_function, step_option):
        """
        The TD(0) algorithm for value function evaluation under the learning setup
        :param policy: The policy to be evaluated
        :param true_value_function: The true value function for the policy (for error estimation)
        :param step_option: can be 1/2/3
        :return: The value function for the policy
        """
        curr_val = np.zeros(self.n_states)
        state_hist = np.zeros(self.n_states)
        s0_error = []
        max_error = []
        print("at TD0:")
        for time_step in tqdm(range(100000)):
            state = np.random.randint(0, self.n_states)
            while state != 0:  # at state = 0 we finish the game, and we do not need to update it because it starts at 0
                next_state, cost = self.simulate_trans(state, policy[state])
                reward = -cost
                d = reward + curr_val[next_state] - curr_val[state]
                state_hist[next_state] += 1
                if step_option == 1:
                    curr_val[state] = curr_val[state] + d / (1 + state_hist[next_state])
                elif step_option == 2:
                    curr_val[state] = curr_val[state] + 0.01 * d
                else:
                    curr_val[state] = curr_val[state] + d * 10 / (100 + state_hist[next_state])
                state = next_state
            s0_error.append(np.abs(true_value_function[-1] - curr_val[-1]))
            max_error.append(np.max(np.abs(true_value_function - curr_val)))

        return curr_val, s0_error, max_error

    def TD_lambda(self, policy, true_value_function, step_option, lam, num_iter=100000):
        """
        The TD(lambda) algorithm for value function evaluation under the learning setup
        :param policy: The policy to be evaluated
        :param true_value_function: The true value function for the policy (for error estimation)
        :param step_option: can be 1/2/3
        :param lam: the lambda parameter for the algorithm
        :return: The value function for the policy
        """
        curr_val = np.zeros(self.n_states)
        state_hist = np.zeros(self.n_states)
        s0_error = []
        max_error = []
        es = np.zeros(self.n_states)
        for time_step in range(num_iter):
            state = np.random.randint(0, self.n_states)
            while state != 0:  # at state = 0 we finish the game, and we do not need to update it because it starts at 0
                next_state, cost = self.simulate_trans(state, policy[state])
                reward = -cost
                d = reward + curr_val[next_state] - curr_val[state]
                state_hist[next_state] += 1
                if step_option == 1:
                    alpha = 1 / (1 + state_hist[next_state])
                elif step_option == 2:
                    alpha = 0.01
                else:
                    alpha = 10 / (100 + state_hist[next_state])
                es = lam * es
                es[state] += 1
                state = next_state
                curr_val = curr_val + alpha * d * es
            s0_error.append(np.abs(true_value_function[-1] - curr_val[-1]))
            max_error.append(np.max(np.abs(true_value_function - curr_val)))

        return curr_val, s0_error, max_error

    def Q_learning(self, step_option, epsilon, optimal_value_function):
        """
        The Q learning algorithm with epsilon greedy exploration for finding the optimal policy
        :param step_option: can be 1/2/3
        :param epsilon: coefficient for the exploration precess
        :param optimal_value_function: for calculating the error of the estimated optimal policy
        :return:
        """
        curr_Q = np.zeros([self.n_states, self.N])
        state_hist = np.zeros(self.n_states)
        s0_error = []
        max_error = []
        print("at Q learning:")
        for step in tqdm(range(100000)):
            state = np.random.randint(0, self.n_states)
            while state != 0:  # at state = 0 we finish the game, and we do not need to update it because it starts at 0
                explore = np.random.rand() < epsilon
                b_state = self._get_binary_state(state)
                allowed_actions = np.where(b_state == 1)[0]
                if explore:
                    action = np.random.choice(allowed_actions)
                else:
                    action = allowed_actions[np.argmax(curr_Q[state, allowed_actions])]
                next_state, cost = self.simulate_trans(state, action)
                reward = -cost
                b_next_state = self._get_binary_state(next_state)
                next_allowed_actions = np.where(b_next_state == 1)[0]
                if next_state == 0:
                    d = d = reward - curr_Q[state, action]
                else:
                    d = reward + np.max(curr_Q[next_state, next_allowed_actions]) - curr_Q[state, action]
                state_hist[state] += 1
                if step_option == 1:
                    curr_Q[state, action] = curr_Q[state, action] + d / (1 + state_hist[state])
                elif step_option == 2:
                    curr_Q[state, action] = curr_Q[state, action] + 0.01 * d
                else:
                    curr_Q[state, action] = curr_Q[state, action] + d * 10 / (100 + state_hist[state])
                state = next_state
            s0_error.append(np.abs(optimal_value_function[-1] - max(curr_Q[-1, :])))
            if step % 100 == 0:
                greedy_policy = np.zeros(self.n_states, dtype=np.uint8)
                for state in range(1, self.n_states):
                    b_state = self._get_binary_state(state)
                    allowed_actions = np.where(b_state == 1)[0]
                    greedy_policy[state] = allowed_actions[np.argmax(curr_Q[state, allowed_actions])]
                V_greedy = self.policy_eval(greedy_policy)
                max_error.append(np.max(np.abs(optimal_value_function - V_greedy)))

        return curr_Q, s0_error, max_error

    def plot_step_sizes(self):
        xs = np.linspace(1, 10000, 10000)
        plt.subplot(131)
        plt.plot(xs, 1 / xs)
        plt.xlabel("n")
        plt.ylabel("f(n) = 1/n(n)")
        plt.subplot(132)
        ys = np.linspace(0.01, 0.01, 10000)
        plt.plot(xs, ys)
        plt.xlabel("n")
        plt.ylabel("f(n) = 0.01")
        plt.subplot(133)
        plt.plot(xs, 10 / (100 + xs))
        plt.xlabel("n")
        plt.ylabel("f(n) = 10/(100+n(n))")
        plt.tight_layout()

    def plot_TD_lambda(self, policy, true_value_function):
        num_rep = 20
        lambdas = [0.1, 0.5, 0.7, 0.8, 0.85]
        colors = ['r', 'g', 'b', 'c', 'k', 'y']
        num_iter_td = 5000
        s0_errors = np.zeros([num_rep, num_iter_td])
        max_errors = np.zeros([num_rep, num_iter_td])
        print("running TD(lambda):")
        for lam_i, lam in enumerate(lambdas):
            print("lambda = {}".format(str(lam)))
            for rep in tqdm(range(num_rep)):
                _, curr_s0, curr_max = self.TD_lambda(policy, true_value_function, 1, lam, num_iter_td)
                s0_errors[rep, :] = curr_s0
                max_errors[rep, :] = curr_max

            mean_s0 = np.mean(s0_errors, axis=0)
            mean_max = np.mean(max_errors, axis=0)
            var_s0 = np.var(s0_errors, axis=0)
            var_max = np.var(max_errors, axis=0)

            # plot
            x = np.linspace(1, num_iter_td, num_iter_td)
            plt.subplot(121)
            plt.plot(x, mean_s0, color=colors[lam_i])
            plt.fill_between(x, (mean_s0 - 1.96 * np.sqrt(var_s0) / np.sqrt(num_rep)),
                             (mean_s0 + 1.96 * np.sqrt(var_s0) / np.sqrt(num_rep)), color=colors[lam_i], alpha=.1)
            plt.subplot(122)
            plt.plot(x, mean_max, color=colors[lam_i])
            plt.fill_between(x, (mean_max - 1.96 * np.sqrt(var_max) / np.sqrt(num_rep)),
                             (mean_max + 1.96 * np.sqrt(var_max) / np.sqrt(num_rep)), color=colors[lam_i], alpha=.1)
        plt.subplot(121)
        plt.xlabel("iteration")
        plt.ylabel("s0 error")
        lam_str = ["$\lambda$ = {}".format(lambdas[i]) for i in range(len(lambdas))]
        plt.legend(lam_str)
        plt.subplot(122)
        plt.xlabel("iteration")
        plt.ylabel("max error")
        plt.legend(lam_str)
        plt.tight_layout()

    def plot_Q_learning(self, optimal_value_function):
        plt.subplot(131)
        _, s0_error, max_error = self.Q_learning(1, 0.1, optimal_value_function)
        plt.plot(s0_error, 'b')
        plt.plot(np.linspace(1, len(s0_error), len(max_error)), max_error, 'r')
        plt.xlabel("time step")
        plt.ylabel("error")
        plt.legend(["s0_error", "max_error"])
        plt.title("$a_n = 1/n(s_n)$")
        plt.subplot(132)
        _, s0_error, max_error = self.Q_learning(2, 0.1, optimal_value_function)
        plt.plot(s0_error, 'b')
        plt.plot(np.linspace(1, len(s0_error), len(max_error)), max_error, 'r')
        plt.xlabel("time step")
        plt.ylabel("error")
        plt.legend(["s0_error", "max_error"])
        plt.title("$a_n = 0.01$")
        plt.subplot(133)
        _, s0_error, max_error = self.Q_learning(3, 0.1, optimal_value_function)
        plt.plot(s0_error, 'b')
        plt.plot(np.linspace(1, len(s0_error), len(max_error)), max_error, 'r')
        plt.xlabel("time step")
        plt.ylabel("error")
        plt.legend(["s0_error", "max_error"])
        plt.title("$a_n = 10/(100+n(s_n))$")


if __name__ == '__main__':
    probs = [0.6, 0.5, 0.3, 0.7, 0.1]
    costs = [1, 4, 6, 2, 9]
    Cmu = Cmu(probs, costs)
    result_path = "./Results/"

    # Planning section:

    # Cost only:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    cost_only_policy = Cmu.get_cost_policy()
    cost_only_value = Cmu.policy_eval(cost_only_policy)
    Cmu.plot_value(cost_only_value, 'o', 'b')
    plt.subplot(122)
    Cmu.plot_policy(cost_only_policy, 'o', 'b')
    plt.savefig(result_path + "Q2_pi_c" + ".png")

    # Cost and Mu policy:
    plt.figure()
    Cmu.plot_value(cost_only_value, 'o', 'b')
    optimal_policy = Cmu.get_cost_mu_policy()
    optimal_value = Cmu.policy_eval(optimal_policy)
    Cmu.plot_value(optimal_value, '*', 'r')

    plt.legend(["cost only policy", "optimal policy"])
    plt.savefig(result_path + "Q2_optimal_vs_cost" + ".png")

    # Policy iter:
    plt.figure()
    policy_iter_res, s_0_val = Cmu.policy_iter()
    plt.plot(s_0_val)
    plt.xlabel("State [decimal representation]")
    plt.ylabel("$V_n(s_o)$")

    plt.savefig(result_path + "Q2_policy_iter" + ".png")

    print("Is the optimal policy and the iter policy the same? {}".format(
        np.array_equal(optimal_policy, policy_iter_res)))

    # Step sizes:
    plt.figure(figsize=(13, 4))
    Cmu.plot_step_sizes()
    plt.savefig(result_path + "Q2_step_sizes" + ".png")

    #  Learning section:
    # TD0_lambda:
    # est_val1, s0_error1, max_error1 = Cmu.TD0(cost_only_policy, cost_only_value, 1)
    # est_val2, s0_error2, max_error2 = Cmu.TD0(cost_only_policy, cost_only_value, 2)
    # est_val3, s0_error3, max_error3 = Cmu.TD0(cost_only_policy, cost_only_value, 3)
    #
    # plt.figure(figsize=(13, 4))
    # plt.subplot(131)
    # plt.plot(s0_error1, 'b')
    # plt.plot(max_error1, 'r')
    # plt.xlabel("time step")
    # plt.ylabel("error")
    # plt.legend(["s0_error", "max_error"])
    # plt.title("$a_n = 1/n(s_n)$")
    #
    # plt.subplot(132)
    # plt.plot(s0_error2, 'b')
    # plt.plot(max_error2, 'r')
    # plt.xlabel("time step")
    # plt.ylabel("error")
    # plt.legend(["s0_error", "max_error"])
    # plt.title("$a_n = 0.01$")
    #
    # plt.subplot(133)
    # plt.plot(s0_error3, 'b')
    # plt.plot(max_error3, 'r')
    # plt.xlabel("time step")
    # plt.ylabel("error")
    # plt.legend(["s0_error", "max_error"])
    # plt.title("$a_n = 10/(100+n(s_n))$")
    # plt.tight_layout()
    #
    # plt.savefig(result_path + "Q2_TD0_error_c" + ".png")

    # TD_lambda:
    # plt.figure(figsize=(10, 4))
    # Cmu.plot_TD_lambda(cost_only_policy, cost_only_value)
    # plt.savefig(result_path + "Q2_TDlambda_error_c" + ".png")

    # Q learning:
    # plt.figure(figsize=(13, 4))
    # Cmu.plot_Q_learning(optimal_value)
    # plt.savefig(result_path + "Q2_Qlearning_error" + ".png")

    plt.figure()
    _, s0_error, max_error = Cmu.Q_learning(2, 0.01, optimal_value)
    plt.plot(s0_error, 'b')
    plt.plot(np.linspace(1, len(s0_error), len(max_error)), max_error, 'r')
    plt.xlabel("time step")
    plt.ylabel("error")
    plt.legend(["s0_error", "max_error"])
    plt.savefig(result_path + "Q2_Qlearning_eps001_error" + ".png")
