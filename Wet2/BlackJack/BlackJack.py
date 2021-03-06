import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pickle


class BlackJack:
    def __init__(self):
        self.value_probs = np.asarray([1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 4 / 13, 1 / 13])
        self.card_values = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        self.n_xs = 17
        self.n_ys = 10
        self.n_states = self.n_xs * self.n_ys + 1

    def _get__exp_reward(self, state, action):
        return 0

    def is_terminal(self, state):
        return state == self.n_states - 1  # The first state is the terminal one

    def ind_to_x_y(self, state):
        y = int(state / self.n_xs + 2)
        x = state - (y - 2) * self.n_xs + 4
        return x, y

    def x_y_to_ind(self, x, y):
        if x >= 21:
            return 171
        return (y - 2) * self.n_xs + x - 4

    def dealers_policy(self, y):
        """
        The dealers policy is to hit while the sum of the cards in his hands are less than 17
        :param y: The current sum of cards in the dealers hands
        :return: hit or stick
        """
        return int(y < 17)

    def get_dealer_prob(self, y, stopy):
        """
        The probability of the dealer to stop at stopy (when starting from y)
        :param y: The starting value
        :param stopy: The stopping value
        :return: The probability
        """

        if y >= 17:
            # The dealer stops
            if stopy > 21:
                # We don't care about the exact value, the dealer is losing in this scenario
                return y > 21
            return y == stopy
        tot_prob = 0
        for i, card_val in enumerate(self.card_values):
            tot_prob += self.value_probs[i] * self.get_dealer_prob(y + card_val, stopy)
        return tot_prob

    def get_r_exp_on_stick(self, state):
        """
        In this stage, the player decided to stick. he has no more actions to do in the game and we just need to
        calculate the expectancy of him to win (based on the dealers policy)
        :param state: The current state of the game (when the player decided to stick)
        :return: the reward expectancy
        """
        x, y = self.ind_to_x_y(state)
        chance_win_stick = 0
        for y_stop in range(y + 2, x):
            chance_win_stick += self.get_dealer_prob(y, y_stop)
        chance_win_stick += self.get_dealer_prob(y, 22)

        chance_draw_stick = self.get_dealer_prob(y, x)
        r_exp_stick = chance_win_stick - (1 - chance_win_stick - chance_draw_stick)
        return r_exp_stick

    def get_r_exp_on_hit(self, state, curr_value):
        exp = 0
        x, y = self.ind_to_x_y(state)
        for i, card in enumerate(self.card_values):
            next_x = x + card
            next_state = self.x_y_to_ind(next_x, y)
            if next_x > 21:
                exp += -1 * self.value_probs[i]
            elif next_x == 21:
                exp += 1 * self.value_probs[i]
            else:
                exp += curr_value[next_state] * self.value_probs[i]
        return exp

    def value_iteration(self, max_num_iter, thresh):
        curr_value = np.zeros(self.n_states)
        curr_policy = np.zeros(self.n_states)
        next_value = np.zeros(self.n_states)

        print("at value iteration:")
        iter = 1
        while True and iter <= max_num_iter:
            print("itreation number: {}".format(iter))
            iter += 1
            for s in range(self.n_states - 1):
                r_exp_stick = self.get_r_exp_on_stick(s)
                r_exp_hit = self.get_r_exp_on_hit(s, curr_value)
                curr_policy[s] = int(r_exp_hit >= r_exp_stick)
                next_value[s] = np.max([r_exp_stick, r_exp_hit])
            # if np.max(np.abs(curr_value - next_value)) <= thresh:
            #     return curr_value, curr_policy
            if np.array_equal(curr_value, next_value):
                return curr_value, curr_policy
            curr_value = next_value.copy()
        return curr_value, curr_policy

    def plot_value_func(self, value_func, fig):

        ax = fig.gca(projection='3d')

        X = np.arange(4, 21, 1)
        Y = np.arange(2, 12, 1)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(list(np.shape(X)))
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                Z[i, j] = value_func[self.x_y_to_ind(X[i, j], Y[i, j])]

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('Player sum of card')
        ax.set_ylabel('Dealers showing')
        ax.set_zlabel('value function')

    def plot_policy(self, policy):
        max_hit_policy = np.full(self.n_ys, self.n_xs)
        for y in range(self.n_ys):
            for x in range(self.n_xs):
                if policy[self.x_y_to_ind(x+4, y+2)]==0:
                    max_hit_policy[y] = x
                    break
        plt.plot(list(range(2, 2 + self.n_ys)), max_hit_policy + 4)
        plt.xticks(list(range(2, 2 + self.n_ys)), list(range(2, 2 + self.n_ys)))
        #plt.grid(which='major',linestyle='--')
        plt.ylim(top=20, bottom=4)
        plt.xlim(right=11, left=2)
        plt.fill_between(list(range(2, 2 + self.n_ys)), max_hit_policy + 4, np.zeros(self.n_ys), color='g')
        plt.fill_between(list(range(2, 2 + self.n_ys)), max_hit_policy + 4, np.full(self.n_ys,self.n_xs+5), color='r')
        plt.text(6, 8, "Hit", size=20,
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )
        plt.text(6, 18, "Stick", size=20,
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )


if __name__ == '__main__':
    results_dir = "./Results/"
    game = BlackJack()
    # optimal_val, optimal_policy = game.value_iteration(50, 0.0001)
    #
    # with open(results_dir + 'value_function.pickle', 'wb') as handle:
    #     pickle.dump(optimal_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(results_dir + 'policy.pickle', 'wb') as handle:
    #     pickle.dump(optimal_policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(results_dir + 'value_function.pickle', 'rb') as handle:
        value = pickle.load(handle)
        fig = plt.figure()
        game.plot_value_func(value, fig)
        # plt.show()
        plt.savefig(results_dir + 'Q1_value_function.png')

    with open(results_dir + 'policy.pickle', 'rb') as handle:
        policy = pickle.load(handle)
        fig = plt.figure()
        game.plot_policy(policy)
        plt.savefig(results_dir + 'Q1_policy.png')
