import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    gamma = np.linspace(0, 0.96, 1000)
    con2 = 1 / (1 - np.power(gamma, 2))
    con6 = 1 / (1 - np.power(gamma, 6))
    con12 = 1 / (1 - np.power(gamma, 12))
    plt.figure()
    plt.plot(gamma, con2)
    plt.plot(gamma, con6)
    plt.plot(gamma, con12)
    plt.xlabel("$\gamma$")
    h = plt.ylabel("$ \\frac{1}{1-\gamma^{2\cdot h}}$")
    h.set_rotation(0)
    plt.legend(["h = 1", "h = 3", "h = 6"])
    plt.savefig("Results/Q3_different_hs.png")
