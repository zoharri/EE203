import numpy as np


def m_prob_word(L):
    prob_table = np.asarray([[0.1, 0.325, 0.25], [0.4, 0, 0.4], [0.2, 0.2, 0.2]])
    cost_table = np.zeros([3, L])
    char_table = np.zeros([3, L])  # This is a table with the actions
    cost_table[:, -1] = [0.325, 0.2, 0.4]
    char_table[:, -1] = [3, 3, 3]  # We want the word to end with '-'
    for k in range(L-2, -1, -1):
        cost1 = np.max(prob_table[0, :] * cost_table[:, k+1])
        char1 = np.argmax(prob_table[0, :] * cost_table[:, k+1])
        cost2 = np.max(prob_table[1, :] * cost_table[:, k+1])
        char2 = np.argmax(prob_table[1, :] * cost_table[:, k+1])
        cost3 = np.max(prob_table[2, :] * cost_table[:, k+1])
        char3 = np.argmax(prob_table[2, :] * cost_table[:, k+1])
        cost_table[:, k] = [cost1, cost2, cost3]
        char_table[:, k] = [char1, char2, char3]
    print(char_table)
    chars = ['B', 'K', 'O']
    word = "B"
    word_idx = np.zeros([L, 1], dtype=int)
    word_idx[0] = 0  # We want the word to start with 'B'
    for k in range(1, L):
        word_idx[k] = char_table[word_idx[k-1], k-1]
        word += chars[int(word_idx[k])]

    return word, cost_table[0, 0]

if __name__ == "__main__":
    word, prob = m_prob_word(5)
    print(word)
    print(prob)