import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer
import pickle
import matplotlib.pyplot as plt


def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):
    """
    Compute the next w given the data.
    :param encoded_states: The current encoded_states
    :param encoded_next_states: The next encoded_states
    :param actions: The actions of the current policy
    :param rewards: The rewards observed
    :param done_flags: Flag if the state is a terminal state
    :param linear_policy: The current policy
    :param gamma: Parameter for the learning process
    :return:  The next w given the data.
    """
    q_features = linear_policy.get_q_features(encoded_states, actions)
    policy_actions = linear_policy.get_max_action(encoded_next_states)
    greedy_q_features = linear_policy.get_q_features(encoded_next_states, policy_actions)
    A = q_features.T @ (q_features - gamma * (greedy_q_features.T * ~done_flags).T) / len(rewards)
    b = q_features.T @ np.expand_dims(rewards, axis=1) / len(rewards)
    return np.linalg.inv(A) @ b


def plot_final_scores(num_smpls):
    """
    plot the final score vs num of samples
    :param num_smpls: list of the number of samples
    :return: None
    """
    plt.figure()
    finals = []
    res_dir = './Results/'
    for smpl in num_smpls:
        with open(res_dir + 'final_perf' + str(int(smpl)) + '.pickle', 'rb') as handle:
            score = pickle.load(handle)
            finals.append(score)
    plt.plot(num_smpls, finals)
    plt.xlabel("Number of samples")
    plt.ylabel("Final success rate")
    plt.savefig(res_dir + "Q2_samples.png")


def plot_perfs():
    """
    plot the mean success rate of the different seeds
    :return: None
    """
    plt.figure()
    res_dir = './Results/'
    with open(res_dir + 'perf' + str(123) + '.pickle', 'rb') as handle:
        per123 = pickle.load(handle)
    with open(res_dir + 'perf' + str(234) + '.pickle', 'rb') as handle:
        per234 = pickle.load(handle)
    with open(res_dir + 'perf' + str(345) + '.pickle', 'rb') as handle:
        per345 = pickle.load(handle)
    with open(res_dir + 'perf' + str(456) + '.pickle', 'rb') as handle:
        per456 = pickle.load(handle)
    with open(res_dir + 'perf' + str(567) + '.pickle', 'rb') as handle:
        per567 = pickle.load(handle)
    mlen = np.min([len(per123), len(per234), len(per345), len(per456), len(per567)])
    av_per = [(per123[i] + per234[i] + per345[i] + per456[i] + per567[i]) / 5 for i in range(mlen)]
    plt.plot(av_per)
    plt.xlabel("Iteration")
    plt.ylabel("Mean success rate")
    plt.savefig(res_dir+"Q2_iter.png")


def run_lspi(seed, w_updates=20, samples_to_collect=100000, evaluation_number_of_games=1,
             evaluation_max_steps_per_game=200, thresh=0.00001, only_final=False):
    """
    This is the main lspi function
    :param seed: random seed for the run
    :param w_updates: how many w updates to do
    :param samples_to_collect: how many samples to collect
    :param evaluation_number_of_games: how many game evaluations to do
    :param evaluation_max_steps_per_game: how many steps to allow the evaluation game to run
    :param thresh: the threshold for the stopping condition
    :param only_final: run evaluation only at the end of the run
    :return: None
    """
    res_dir = './Results/'
    np.random.seed(seed)
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.999
    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    # get data success rate
    data_success_rate = np.sum(rewards) / len(rewards)
    print('success rate: {}'.format(data_success_rate))
    # standardize data
    data_transformer = DataTransformer()
    data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
    states = data_transformer.transform_states(states)
    next_states = data_transformer.transform_states(next_states)
    # process with radial basis functions
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    # encode all states:
    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
    encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
    # set a new linear policy
    linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
    # but set the weights as random
    linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
    # start an object that evaluates the success rate over time
    evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)

    # success_rate = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
    # print("Initial success rate: {}".format(success_rate))
    performances = []
    if not only_final:
        performances.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
    read = False
    if read:
        with open(res_dir + 'weight.pickle', 'rb') as handle:
            new_w = pickle.load(handle)
            linear_policy.set_w(np.expand_dims(new_w, 1))
    for lspi_iteration in range(w_updates):
        print('starting lspi iteration {}'.format(lspi_iteration))

        new_w = compute_lspi_iteration(
            encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
        )
        with open(res_dir + 'weight.pickle', 'wb') as handle:
            pickle.dump(new_w, handle, protocol=pickle.HIGHEST_PROTOCOL)

        norm_diff = linear_policy.set_w(new_w)
        if not only_final:
            performances.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
        if norm_diff < thresh:
            break
    print('done lspi')
    if not only_final:
        with open(res_dir + 'perf' + str(seed) + '.pickle', 'wb') as handle:
            pickle.dump(performances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if only_final:
        score = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
        with open(res_dir + 'final_perf' + str(samples_to_collect) + '.pickle', 'wb') as handle:
            pickle.dump(score, handle, protocol=pickle.HIGHEST_PROTOCOL)
    evaluator.play_game(evaluation_max_steps_per_game, render=True)


if __name__ == '__main__':
    # run_lspi(123, samples_to_collect=100000, w_updates=10, thresh=0, evaluation_number_of_games=10)
    # run_lspi(234, samples_to_collect=100000, w_updates=10, thresh=0, evaluation_number_of_games=10)
    # run_lspi(345, samples_to_collect=100000, w_updates=10, thresh=0, evaluation_number_of_games=10)
    # run_lspi(456, samples_to_collect=100000, w_updates=10, thresh=0, evaluation_number_of_games=10)
    # run_lspi(567, samples_to_collect=100000, w_updates=10, thresh=0, evaluation_number_of_games=10)
    plot_perfs()

    # num_smpls = np.linspace(10, 300000, 20)
    # for smpl in num_smpls:
    #     run_lspi(345, samples_to_collect=int(smpl), w_updates=20, evaluation_number_of_games=20, only_final=True)
    # plot_final_scores(num_smpls)
    run_lspi(123, samples_to_collect=100000, w_updates=10, thresh=0.00001, evaluation_number_of_games=1)

