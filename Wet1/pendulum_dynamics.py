import numpy as np


def get_A_iLQR(cart_pole_env, x_t, u_t):
    '''
    create and returns the A matrix used in iLQR, this is the x term in the linearization of the pendulum dynamics
    (see the pdf for details)
    :param cart_pole_env: to extract all the relevant constants
    :param x_t: the x value to do linearization around
    :param u_t: the u value to do linearization around
    :return: the A matrix used in iLQR
    '''
    g = cart_pole_env.gravity
    m = cart_pole_env.masspole
    M = cart_pole_env.masscart
    l = cart_pole_env.length
    dt = cart_pole_env.tau
    stheta_t = np.asscalar(np.sin(x_t[2]))
    ctheta_t = np.asscalar(np.cos(x_t[2]))
    dtheta_t = np.asscalar(x_t[3])
    u_t = np.asscalar(u_t)

    denom_xdd = (M+m-m*ctheta_t**2)
    # print("s:{}".format(stheta_t))
    # print("c:{}".format(ctheta_t))
    # print("d:{}".format(dtheta_t))
    #
    # print((u_t+m*l*stheta_t*dtheta_t**2-m*g*ctheta_t*stheta_t)/denom_xdd)

    free_taylor_x = (u_t+m*l*stheta_t*dtheta_t**2-m*g*ctheta_t*stheta_t)/denom_xdd
    theta_taylor1_x =(g*m*stheta_t**2-g*m*ctheta_t**2+l*m*dtheta_t**2*ctheta_t)/denom_xdd
    theta_taylor2_x = (2*m*stheta_t*ctheta_t*(-g*m*stheta_t*ctheta_t+l*m*dtheta_t**2*stheta_t+u_t))/denom_xdd
    theta_taylor_x = theta_taylor1_x-theta_taylor2_x
    theta_dot_taylor_x = (2*l*m*dtheta_t*stheta_t)/denom_xdd
    u_taylor_x = 1/denom_xdd

    denom_tdd = m*l*ctheta_t**2-(M+m)*l
    free_taylor_t = (u_t*ctheta_t-(M+m)*g*stheta_t+m*l*ctheta_t*stheta_t*dtheta_t**2)/denom_tdd
    theta_taylor1_t = (-dtheta_t**2*l*m*stheta_t**2+dtheta_t**2*l*m*ctheta_t**2-g*(m+M)*ctheta_t-u_t*stheta_t)/denom_tdd
    theta_taylor2_t =(2*l*m*stheta_t*ctheta_t*(dtheta_t**2*l*m*stheta_t*ctheta_t-g*(m+M)*stheta_t+u_t*ctheta_t))/(denom_tdd**2)
    theta_taylor_t = theta_taylor1_t + theta_taylor2_t
    theta_dot_taylor_t = -(2*m*dtheta_t*stheta_t*ctheta_t)/(-m*ctheta_t**2+m+M)
    u_taylor_t = ctheta_t/denom_tdd


    A_bar = np.array([[0, 1, 0, 0 ]\
                    , [0, 0, -theta_taylor_x, theta_dot_taylor_x]\
                    , [0, 0, 0, 1]\
                    , [0, 0, theta_taylor_t, theta_dot_taylor_t]])

    return np.eye(4) + dt*A_bar


def get_B_iLQR(cart_pole_env, x_t, u_t):
    '''
    create and returns the B matrix used in iLQR, this is the u term in the linearization of the pendulum dynamics
    (see the pdf for details)
    :param cart_pole_env: to extract all the relevant constants
    :param x_t: the x value to do linearization around
    :param u_t: the u value to do linearization around
    :return: the A matrix used in iLQR
    '''
    g = cart_pole_env.gravity
    m = cart_pole_env.masspole
    M = cart_pole_env.masscart
    l = cart_pole_env.length
    dt = cart_pole_env.tau
    stheta_t = np.asscalar(np.sin(x_t[2]))
    ctheta_t = np.asscalar(np.cos(x_t[2]))
    dtheta_t = np.asscalar(x_t[3])
    u_t = np.asscalar(u_t)

    denom_xdd = (M + m - m * ctheta_t ** 2)
    free_taylor_x = (u_t + m * l * stheta_t * dtheta_t ** 2 - m * g * ctheta_t * stheta_t) / denom_xdd
    theta_taylor1_x = (g * m * stheta_t ** 2 - g * m * ctheta_t ** 2 + l * m * dtheta_t ** 2 * ctheta_t) / denom_xdd
    theta_taylor2_x = (2 * m * stheta_t * ctheta_t * (
                -g * m * stheta_t * ctheta_t + l * m * dtheta_t ** 2 * stheta_t + u_t)) / denom_xdd
    theta_taylor_x = theta_taylor1_x - theta_taylor2_x
    theta_dot_taylor_x = (2 * l * m * dtheta_t * stheta_t) / denom_xdd
    u_taylor_x = 1 / denom_xdd

    denom_tdd = m * l * ctheta_t ** 2 - (M + m) * l
    free_taylor_t = (u_t * ctheta_t - (M + m) * g * stheta_t + m * l * ctheta_t * stheta_t * dtheta_t ** 2) / denom_tdd
    theta_taylor1_t = (-dtheta_t ** 2 * l * m * stheta_t ** 2 + dtheta_t ** 2 * l * m * ctheta_t ** 2 - g * (
                m + M) * ctheta_t - u_t * stheta_t) / denom_tdd
    theta_taylor2_t = (2 * l * m * stheta_t * ctheta_t * (
                dtheta_t ** 2 * l * m * stheta_t * ctheta_t - g * (m + M) * stheta_t + u_t * ctheta_t)) / (
                                  denom_tdd ** 2)
    theta_taylor_t = theta_taylor1_t + theta_taylor2_t
    theta_dot_taylor_t = -(2 * m * dtheta_t * stheta_t * ctheta_t) / (-m * ctheta_t ** 2 + m + M)
    u_taylor_t = ctheta_t / denom_tdd

    B_bar = np.array([[0,u_taylor_x, 0, -u_taylor_t]])
    return (B_bar*dt).T


def get_D_iLQR(cart_pole_env, x_t, u_t):
    '''
    create and returns the D matrix used in iLQR, this is the free term in the linearization of the pendulum dynamics
    (see the pdf for details)
    :param cart_pole_env: to extract all the relevant constants
    :param x_t: the x value to do linearization around
    :param u_t: the u value to do linearization around
    :return: the A matrix used in iLQR
    '''
    g = cart_pole_env.gravity
    m = cart_pole_env.masspole
    M = cart_pole_env.masscart
    l = cart_pole_env.length
    dt = cart_pole_env.tau
    stheta_t = np.sin(x_t[2])
    ctheta_t = np.cos(x_t[2])
    dtheta_t = x_t[3]

    denom_xdd = (M + m - m * ctheta_t ** 2)
    free_taylor_x = (u_t + m * l * stheta_t * dtheta_t ** 2 - m * g * ctheta_t * stheta_t) / denom_xdd
    theta_taylor1_x = (g * m * stheta_t ** 2 - g * m * ctheta_t ** 2 + l * m * dtheta_t ** 2 * ctheta_t) / denom_xdd
    theta_taylor2_x = (2 * m * stheta_t * ctheta_t * (
                -g * m * stheta_t * ctheta_t + l * m * dtheta_t ** 2 * stheta_t + u_t)) / denom_xdd
    theta_taylor_x = theta_taylor1_x - theta_taylor2_x
    theta_dot_taylor_x = (2 * l * m * dtheta_t * stheta_t) / denom_xdd
    u_taylor_x = 1 / denom_xdd

    denom_tdd = m * l * ctheta_t ** 2 - (M + m) * l
    free_taylor_t = (u_t * ctheta_t - (M + m) * g * stheta_t + m * l * ctheta_t * stheta_t * dtheta_t ** 2) / denom_tdd
    theta_taylor1_t = (-dtheta_t ** 2 * l * m * stheta_t ** 2 + dtheta_t ** 2 * l * m * ctheta_t ** 2 - g * (
                m + M) * ctheta_t - u_t * stheta_t) / denom_tdd
    theta_taylor2_t = (2 * l * m * stheta_t * ctheta_t * (
                dtheta_t ** 2 * l * m * stheta_t * ctheta_t - g * (m + M) * stheta_t + u_t * ctheta_t)) / (
                                  denom_tdd ** 2)
    theta_taylor_t = theta_taylor1_t + theta_taylor2_t
    theta_dot_taylor_t = -(2 * m * dtheta_t * stheta_t * ctheta_t) / (-m * ctheta_t ** 2 + m + M)
    u_taylor_t = ctheta_t / denom_tdd

    #free_xdd = free_taylor_x-x_t[2]*theta_taylor_t-x_t[3]*theta_dot_taylor_t-u_t*u_taylor_t
    D_bar = np.array([[0, free_taylor_x, 0, free_taylor_t]])
    return (D_bar*dt).T


def get_A_LQR(cart_pole_env):
    '''
    create and returns the A matrix used in LQR, this is the x term in the linearization of the pendulum dynamics
    (see the pdf for details)
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in iLQR
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    A_bar = np.array([[0, 1, 0, 0], [0, 0, pole_mass*g/cart_mass, 0], [0, 0, 0, 1], [0, 0, (1+pole_mass/cart_mass)*g/pole_length, 0]])

    return np.eye(4) + dt*A_bar


def get_B_LQR(cart_pole_env):
    '''
    create and returns the B matrix used in LQR, this is the u term in the linearization of the pendulum dynamics
    (see the pdf for details)
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in iLQR
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    B_bar = np.array([[0, 1/cart_mass, 0, 1/(cart_mass*pole_length)]])

    return (B_bar*dt).T
