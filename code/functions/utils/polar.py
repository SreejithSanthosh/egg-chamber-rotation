"""Functions for working with vectors and polar coordinates."""

import numpy as np


def cart_to_pol(x, y):
    """
    Convert vector from cartesian/xy notation to polar notation.

    Parameters
    ------
    x: int or float
        The x component of a vector
    y: int or float
        The y component of a vector

    Returns
    -------
    phi: float
        The direction of the vector
    rho: float
        The magnitude of the vector
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def pol_to_cart(phi, rho=1):
    """
    Convert vector from polar notation to cartesian/xy notation.

    Parameters
    -------
    phi: float
        The direction of the vector
    rho: float
        The magnitude of the vector.
        If ommitted, the default value is 1 (unit vector).

    Returns
    ------
    x: int or float
        The x component of a vector
    y: int or float
        The y component of a vector
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def points_to_angle(p, q):
    """
    Finds the polar angle (phi) between the vector from the origin
    along the positive x-axis and the line defined by two points p,q.

    Parameters
    -----
    p: tuple of ints/floats. (x,y)
    q: tuple of ints/floats. (x,y)

    Returns
    ------
    phi: float
        the angle of the line in radians, between -pi and pi.
    """
    phi = np.arctan2(q[0] - p[0], q[1] - p[1])
    return phi


def wrap_to_pi(x):
    """
    Wrap a value between -pi and pi in polar coordinate space.
        e.g. if x = 2*pi, x_wrapped = 0
        e.g. if x = 3*pi, x_wrapped = pi

    Parameters
    ------
    x: an int or float

    Returns
    -------
    x_wrapped: float between -pi and pi
    """
    x_wrapped = (x + np.pi) % (2 * np.pi) - np.pi
    return x_wrapped
