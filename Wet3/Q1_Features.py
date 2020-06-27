from Wet3.radial_basis_function_extractor import RadialBasisFunctionExtractor
from Wet3.mountain_car_with_data_collection import MountainCarWithResetEnv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_two_radial(RBFE:RadialBasisFunctionExtractor, feature_num, fig):

    ax = fig.add_subplot(1, 2, feature_num+1, projection='3d')
    # Make data
    xs = np.linspace(-2.5, 2.5, 100)
    ys = np.linspace(-2.5, 2.5, 100)
    xv, yv = np.meshgrid(xs, ys)
    z = np.zeros([len(xs), len(ys)])
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            state = [[x, y]]
            #state = np.expand_dims(state, axis=0)
            z[ix, iy] = RBFE.encode_states_with_radial_basis_functions(state)[0][feature_num]
    # Plot the surface
    ax.plot_surface(xv, yv, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Position of Car', fontsize=20)
    ax.set_ylabel('Velocity of Car', fontsize=20)
    ax.set_zlabel('Feature Number: {}'.format(feature_num+1))
    ax.view_init(elev=40., azim=240)

if __name__ == '__main__':
    resdir = "./Results/"
    RBFE = RadialBasisFunctionExtractor([12, 10])
    game = MountainCarWithResetEnv()
    state_space = game.observation_space
    low = state_space.low
    high = state_space.high
    fig = plt.figure(figsize=(13, 4))
    plot_two_radial(RBFE, 0, fig)
    plot_two_radial(RBFE, 1, fig)
    plt.savefig(resdir + "Q1_features.png")
