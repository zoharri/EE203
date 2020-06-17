from Wet3.radial_basis_function_extractor import RadialBasisFunctionExtractor
from Wet3.mountain_car_with_data_collection import MountainCarWithResetEnv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_two_radial(RBFE:RadialBasisFunctionExtractor, low, high):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Make data
    xs = np.linspace(low[0], high[0], 100)
    ys = np.linspace(low[1], high[1], 100)
    xv, yv = np.meshgrid(xs, ys)
    z = np.zeros([len(xs), len(ys)])
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            z[ix, iy] = RBFE.encode_states_with_radial_basis_functions([[x, y]])[0][3]

    # Plot the surface
    ax.plot_surface(xv, yv, z, color='b')
    plt.show()

if __name__ == '__main__':
    RBFE = RadialBasisFunctionExtractor([10, 10])
    game = MountainCarWithResetEnv()
    state_space = game.observation_space
    low = state_space.low
    high = state_space.high
    plot_two_radial(RBFE, low, high)
