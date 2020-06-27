import numpy as np
import time

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor
import pickle
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, number_of_kernels_per_dim, number_of_actions, gamma, learning_rate, hard_theta=False):
        # Set max value for normalization of inputs
        self._max_normal = 1
        # get state \action information
        self.data_transformer = DataTransformer()
        state_mean = [-3.00283763e-01, 5.61618575e-05]
        state_std = [0.51981243, 0.04024895]
        self.data_transformer.set(state_mean, state_std)
        self._actions = number_of_actions
        # create RBF features:
        self.feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        self.number_of_features = self.feature_extractor.get_number_of_features()
        # the weights of the q learner
        if hard_theta:
            self.theta = np.random.uniform(-10, 10, size=number_of_actions * self.number_of_features)
        else:
            self.theta = np.random.uniform(-0.001, 0, size=number_of_actions * self.number_of_features)
        # discount factor for the solver
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _normalize_state(self, s):
        return self.data_transformer.transform_states(np.array([s]))[0]

    def get_features(self, state):
        normalized_state = self._normalize_state(state)
        features = self.feature_extractor.encode_states_with_radial_basis_functions([normalized_state])[0]
        return features

    def get_q_val(self, features, action):
        theta_ = self.theta[action * self.number_of_features: (1 + action) * self.number_of_features]
        return np.dot(features, theta_)

    def get_all_q_vals(self, features):
        all_vals = np.zeros(self._actions)
        for a in range(self._actions):
            all_vals[a] = solver.get_q_val(features, a)
        return all_vals

    def get_max_action(self, state):
        sparse_features = solver.get_features(state)
        q_vals = solver.get_all_q_vals(sparse_features)
        return np.argmax(q_vals)

    def get_state_action_features(self, state, action):
        state_features = self.get_features(state)
        all_features = np.zeros(len(state_features) * self._actions)
        all_features[action * len(state_features): (1 + action) * len(state_features)] = state_features
        return all_features

    def update_theta(self, state, action, reward, next_state, done):
        # compute the new weights and set in self.theta. also return the bellman error (for tracking).
        max_action = self.get_max_action(next_state)
        max_next_q = self.get_q_val(self.get_features(next_state), max_action)
        curr_feat = self.get_features(state)
        curr_q = self.get_q_val(curr_feat, action)
        d = reward + int(not done) * max_next_q * self.gamma - curr_q
        curr_ac_feat = self.get_state_action_features(state, action)
        new_theta = self.theta + self.learning_rate * d * curr_ac_feat
        self.theta = new_theta
        return d ** 2


def modify_reward(reward):
    reward -= 1
    if reward == 0.:
        reward = 100.
    return reward


def run_episode(env, solver, is_train=True, epsilon=None, max_steps=200, render=False, bonus=False):
    episode_gain = 0
    deltas = []
    if is_train:
        if bonus:
            start_position = -0.5
            start_velocity = np.random.uniform(0, 0)
            # start_position = np.random.uniform(env.min_position, env.goal_position - 0.01)
            # start_velocity = np.random.uniform(-env.max_speed, env.max_speed)
        else:
            start_position = np.random.uniform(env.min_position, env.goal_position - 0.01)
            start_velocity = np.random.uniform(-env.max_speed, env.max_speed)
    else:
        start_position = -0.5
        start_velocity = np.random.uniform(-env.max_speed / 100., env.max_speed / 100.)
    state = env.reset_specific(start_position, start_velocity)
    step = 0
    if render:
        env.render()
        time.sleep(0.1)
    while True:
        if epsilon is not None and np.random.uniform() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = solver.get_max_action(state)
        if render:
            env.render()
            time.sleep(0.1)
        next_state, reward, done, _ = env.step(action)
        reward = modify_reward(reward)
        step += 1
        episode_gain += reward
        if is_train:
            deltas.append(solver.update_theta(state, action, reward, next_state, done))
        if done or step == max_steps:
            return episode_gain, np.mean(deltas)
        state = next_state


def run_q_learning(seed, solver, bonus=False, hard_theta=False, eps_comp=False, ep_idx=0, eps=0.1):
    """
    Main function for the online q learning
    :param seed: random seed
    :return: None
    """

    np.random.seed(seed)
    env.seed(seed)
    res_dir = './Results/'

    epsilon_current = eps
    epsilon_decrease = 1.
    epsilon_min = 0.05

    # max_episodes = 30
    max_episodes = 500

    best_gain = -np.infty
    train_gains = []
    mean_test_gains = []
    Bellman_errors = []
    bottom_value = []
    for episode_index in range(1, max_episodes + 1):
        episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current, bonus=bonus)
        train_gains.append(episode_gain)
        Bellman_errors.append(mean_delta)
        # reduce epsilon if required
        epsilon_current *= epsilon_decrease
        epsilon_current = max(epsilon_current, epsilon_min)
        bottom_value.append(np.max(solver.get_all_q_vals(solver.get_features(np.array([-0.5, 0])))))

        print('after {}, reward = {}, epsilon {}, average error {}'.format(episode_index, episode_gain, epsilon_current,
                                                                           mean_delta))

        # termination condition:
        if episode_index % 10 == 9:
            test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
            mean_test_gain = np.mean(test_gains)
            mean_test_gains.append(mean_test_gain)
            if best_gain <= mean_test_gain:
                with open(res_dir + 'Q3_weight1.pickle', 'wb') as handle:
                    pickle.dump(solver.theta, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('tested 10 episodes: mean gain is {}'.format(mean_test_gain))
            if mean_test_gain >= -75.:
                print('solved in {} episodes'.format(episode_index))
                break
    save_dir = res_dir
    if bonus:
        save_dir = res_dir + 'Bonus/'
        if hard_theta:
            save_dir += 'hard_thata_'
    if eps_comp:
        save_dir += "epscomp_" +str(ep_idx)
    with open(save_dir + 'Q_learning_mean_test_gains' + str(seed) + '.pickle', 'wb') as handle:
        pickle.dump(mean_test_gains, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_dir + 'Q_learning_train_gains' + str(seed) + '.pickle', 'wb') as handle:
        pickle.dump(train_gains, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_dir + 'Q_learning_bellman_errors' + str(seed) + '.pickle', 'wb') as handle:
        pickle.dump(Bellman_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(save_dir + 'Q_learning_bottom_value' + str(seed) + '.pickle', 'wb') as handle:
        pickle.dump(bottom_value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # run_episode(env, solver, is_train=False, render=True)


def plot_graphs_3_sep_eps(seed, eps):
    res_dir = './Results/'

    # plot training rewards:
    legend_strs = [str(ep) for ep in eps]
    plt.figure(figsize=(12, 7))
    for i, _ in enumerate(eps):
        res_dir = './Results/'
        res_dir += "epscomp_" + str(i)

        with open(res_dir + 'Q_learning_train_gains' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            box = np.ones(10) / 10
            vec_smooth = np.convolve(vec, box, mode='same')
            plt.plot(vec_smooth)

    plt.suptitle("Seed = {}".format(seed))
    plt.legend(legend_strs)
    plt.xlabel("Iteration")
    plt.ylabel("Total training gain")
    res_dir = './Results/epscomp_'
    plt.savefig(res_dir + 'Q_learning_res_{}.png'.format(str(seed)))


def plot_graphs_3_sep(seeds, bonus=False, hard_theta=False, eps_comp=False, ep_idx=0):
    res_dir = './Results/'
    if bonus:
        res_dir += 'Bonus/'
    if eps_comp:
        res_dir += "epscomp_" + str(ep_idx)

    # plot training rewards:

    for seed in seeds:
        plt.figure(figsize=(12, 7))
        plt.subplot(221)
        with open(res_dir + 'Q_learning_mean_test_gains' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            plt.plot(10 * np.asarray(range(len(vec))), vec)
            plt.xlabel("Iteration")
            plt.ylabel("Mean test gain")
        plt.subplot(222)
        with open(res_dir + 'Q_learning_train_gains' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            plt.plot(vec)
            plt.xlabel("Iteration")
            plt.ylabel("Total training gain")
        plt.subplot(223)
        with open(res_dir + 'Q_learning_bottom_value' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            plt.plot(vec)
            plt.xlabel("Iteration")
            plt.ylabel("Value function of bottom state")
        plt.subplot(224)
        with open(res_dir + 'Q_learning_bellman_errors' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            box = np.ones(10) / 10
            vec_smooth = np.convolve(vec, box, mode='same')
            plt.plot(vec_smooth)
            plt.xlabel("Iteration")
            plt.ylabel("Bellman error (smoothed)")
        plt.suptitle("Seed = {}".format(seed))
        save_dir = res_dir
        if bonus:
            save_dir = save_dir + 'Bonus_'
            if hard_theta:
                save_dir += 'hard_thata_'
        plt.savefig(save_dir + 'Q_learning_res_{}.png'.format(str(seed)))


def plot_graphs_3(seeds):
    res_dir = './Results/'
    legend_strs = ["seed = {}".format(str(seed)) for seed in seeds]
    # plot training rewards:
    plt.figure()
    for seed in seeds:
        with open(res_dir + 'Q_learning_mean_test_gains' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            plt.plot(vec)
    plt.xlabel("Iteration")
    plt.ylabel("Mean test gain")
    plt.legend(legend_strs)
    plt.savefig(res_dir + 'Q_learning_mean_test_gains.png')

    # plot testing rewards:
    plt.figure()
    for seed in seeds:
        with open(res_dir + 'Q_learning_train_gains' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            plt.plot(vec)
    plt.xlabel("Iteration")
    plt.ylabel("Total training gain")
    plt.legend(legend_strs)
    plt.savefig(res_dir + 'Q_learning_train_gains.png')

    # plot value of state:
    plt.figure()
    for seed in seeds:
        with open(res_dir + 'Q_learning_bottom_value' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            plt.plot(vec)
    plt.xlabel("Iteration")
    plt.ylabel("Value function of bottom state")
    plt.legend(legend_strs)
    plt.savefig(res_dir + 'Q_learning_bottom_value.png')

    # plot total bellman error:
    plt.figure()
    for seed in seeds:
        with open(res_dir + 'Q_learning_bellman_errors' + str(seed) + '.pickle', 'rb') as handle:
            vec = pickle.load(handle)
            box = np.ones(10) / 10
            vec_smooth = np.convolve(vec, box, mode='same')
            plt.plot(vec_smooth)
    plt.xlabel("Iteration")
    plt.ylabel("Bellman error (smoothed)")
    plt.legend(legend_strs)
    plt.savefig(res_dir + 'Q_learning_bellman_errors.png')


if __name__ == "__main__":
    gamma = 0.999
    env = MountainCarWithResetEnv()
    learning_rate = 0.01
    hard_theta = False
    bonus = False
    eps_comp = True
    solver = Solver(
        # learning parameters
        gamma=gamma, learning_rate=learning_rate,
        # feature extraction parameters
        number_of_kernels_per_dim=[7, 5],
        # env dependencies (DO NOT CHANGE):
        number_of_actions=env.action_space.n,
        hard_theta=hard_theta
    )
    seeds = [789]
    if eps_comp:
        eps = [0.01, 0.3, 0.5, 0.75, 1]
        # eps = [0.1]
        for seed in seeds:
            print("at seed: {}".format(str(seed)))
            for i in range(len(eps)):
                run_q_learning(seed, solver, bonus, hard_theta, eps_comp, i, eps[i])
            plot_graphs_3_sep_eps(seed, eps)
    else:
        for seed in seeds:
            print("at seed: {}".format(str(seed)))
            run_q_learning(seed, solver, bonus, hard_theta, eps_comp)
        plot_graphs_3_sep(seeds, bonus, hard_theta, eps_comp)
