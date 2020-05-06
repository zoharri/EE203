import numpy as np
from cartpole_cont import CartPoleContEnv
import matplotlib.pyplot as plt
from pendulum_dynamics import get_A_iLQR as get_A
from pendulum_dynamics import get_B_iLQR as get_B
from pendulum_dynamics import get_D_iLQR as get_D


"""
# This is not Working :(
"""


def find_lqr_for_ilqr(cart_pole_env, xs_prev, us_prev, limited_force = False):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    xs_prev, us_prev - rollout to linearize the dynamics around
    limited_force - is the force limited in the environment
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)
    w1 = 0.9
    w2 = 0.5
    w3 = 0.1
    if limited_force:
        w3 = 0.5

    C = np.array([
        [w1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, w2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, w3]
    ])
    A =  get_A()
    B =  get_B()
    D =  get_D()
    F = np.concatenate((A,B),1)
    f = D


    us = []
    xs = [xs_prev[0]]

    Ks = []
    Ps.append(Q)

    for step in range(cart_pole_env.planning_steps):
        A = get_A(cart_pole_env, xs_prev[-step], us_prev[-step])
        B = get_B(cart_pole_env, xs_prev[-step], us_prev[-step])

        curr_P = Q+A.T@Ps[-1]@A-A.T@Ps[-1]@B*np.reciprocal(R+B.T@Ps[-1]@B)*B.T@Ps[-1]@A
        curr_K = -np.reciprocal(B.T@Ps[-1]@B+R)*B.T@Ps[-1]@A
        Ps.append(curr_P)
        Ks.append(curr_K)

    Ks.reverse()
    Ps.reverse()
    for step in range(cart_pole_env.planning_steps):
        curr_u = Ks[step]@(xs[-1]-xs_prev[step]) + us_prev[step]
        A = get_A(cart_pole_env, xs[-1], curr_u)
        B = get_B(cart_pole_env, xs[-1], curr_u)
        next_X = A@xs[-1]+B*curr_u
        next_X[2] = np.mod(next_X[2] + np.pi, 2 * np.pi) - np.pi
        xs.append(next_X)
        us.append(curr_u)


    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def find_ilqr_control_input(cart_pole_env, limited_force = False, n_iter = 100):
    '''
    implements the iLQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    # Initialize us with the LQR output at the initial state
    xs = [np.zeros((4, 1)) for i in range(cart_pole_env.planning_steps + 1)]
    us = [np.zeros(1) for i in range(cart_pole_env.planning_steps)]

    for iter in range(n_iter):
        print("at ILQR iter: {} ".format(str(iter)))
        xs, us, Ks = find_lqr_for_ilqr(cart_pole_env, xs, us, limited_force)

    #xs, us, Ks = find_lqr_for_ilqr(cart_pole_env, xs_0, us_0, limited_force)

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


def excecute_iLQR(init_theta, use_predicted, limited_force=False):
    curr_theta_values = []
    env = CartPoleContEnv(initial_theta=init_theta)


    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_ilqr_control_input(env, limited_force)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] @ np.expand_dims(actual_state, 1)).item(0)
        if use_predicted:
            actual_action = predicted_action
        # print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        env.render()
        iteration += 1
        curr_theta_values.append(np.mod(actual_theta + np.pi, 2 * np.pi) - np.pi)
    times = np.arange(0.0, env.tau*len(curr_theta_values), env.tau)
    env.close()
    return curr_theta_values, times


def plot_thetas(thetas, use_predicted, limited_force=False):
    theta_values = []
    ts = []
    for theta in thetas:
        curr_theta_values, curr_ts = excecute_LQR(theta, use_predicted, limited_force)
        ts.append(curr_ts)
        theta_values.append(curr_theta_values)

    # plot
    fig, ax = plt.subplots()
    ax.set(xlabel='t [s]', ylabel='pole angle [rad]')
           #title='Plotting of the actual pole angle with different initial angles')
    ax.grid()
    for theta_idx, theta in enumerate(thetas):
        ax.plot(ts[theta_idx], theta_values[theta_idx])
    thetas_strs = ["init angle = {}".format(str(np.round(theta, 4)))for theta in thetas]
    plt.legend(thetas_strs)
    if use_predicted:
        fig.savefig("Wet1/graphs/unstable_theta_predicted.png")
    else:
        if limited_force:
            fig.savefig("Wet1/graphs/unstable_theta_lim_force.png")
        else:
            fig.savefig("Wet1/graphs/unstable_theta.png")
    plt.show()


if __name__ == '__main__':
    unstable_theta = 0.33*np.pi
    #unstable_theta = 0.05 * np.pi
    #plot_thetas([0.05 * np.pi, 0.005 * np.pi, 0.0005 * np.pi], True)
    #plot_thetas([0.1 * np.pi, unstable_theta, 0.5*unstable_theta], False)
    excecute_iLQR(0.4*np.pi, False, False)
    #plot_thetas([0.1 * np.pi, unstable_theta, 0.5 * unstable_theta], False, False)

    """
    env = CartPoleContEnv(initial_theta=np.pi * 0.323)
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.25)

    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] @ np.expand_dims(actual_state, 1)).item(0)
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))
    """

