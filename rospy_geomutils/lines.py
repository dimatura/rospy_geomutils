"""
manipulate 2D lines in image space.
TODO: refactor related utilities in parafina
"""

from __future__ import division
import numpy as np
from .euclid import Point2, Line2


__all__ = [
           'abc_to_center_abc',
           'abc_to_center_rhotheta',
           'abc_to_endpoints',
           'abc_to_rhotheta',
           'clip_line',
           'endpoints_to_abc',
           'endpoints_to_center_rhotheta',
           'endpoints_to_rhotheta',
           'fix_endpoints',
           'rhotheta_to_center_endpoints',
           'rhotheta_to_endpoints',
           ]


def endpoints_to_abc(xs, ys):
    """ given line segment as endpoint,
    return coeffs ax + by + c = 0
    """
    eps = 1e-8
    x0, x1 = xs
    y0, y1 = ys
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - y0*x1
    norm = np.sqrt(a**2 + b**2) + eps
    a /= norm
    b /= norm
    c /= norm
    return a, b, c


def abc_to_endpoints(a, b, c, w, h):
    """ given coeffs ax + by + c = 0, and img size,
    return line segment endpoints.
    TODO: check for unit norm?
    """
    eps = 1e-8
    if (np.abs(a) < eps) and (np.abs(b) < eps):
        raise ValueError('both a and b are near 0')

    norm = np.sqrt(a**2 + b**2)
    if not np.allclose(norm, 1.0, atol=1e-6):
        raise ValueError('(a,b) must have unit norm')

    if np.abs(b) < eps:
        # vertical line
        xs = [-c/a, -c/a]
        ys = [0., h]
        return xs, ys

    assert(np.abs(b) > eps)

    # ax + by + c = 0 | x=0
    # by + c = 0
    # y = -c/b
    p0 = Point2(0., -c/b)
    # ax + by + c = 0 | x=w
    # aw + by + c = 0
    # y = -(c+aw)/b
    p1 = Point2(w, -(c+a*w)/b)

    ln = Line2(p0, p1)

    top_left     = Point2(0., 0.)
    top_right    = Point2(w , 0.)
    bottom_left  = Point2(0., h)
    bottom_right = Point2(w , h)

    #left_edge = Line2(top_left, bottom_left)
    #right_edge = Line2(top_right, bottom_right)
    top_edge = Line2(top_left, top_right)
    bottom_edge = Line2(bottom_left, bottom_right)

    # if line is completely above or below image, bail
    if ((p0.y < 0 and p1.y < 0) or
        (p0.y >= h and p1.y >= h)):
        xs = [p0.x, p1.x]
        ys = [p0.y, p1.y]
        return xs, ys

    # if only one of the endpoints is above/below image,
    # clip to top/bottom

    if p0.y < 0:
        p0 = ln.intersect(top_edge)
    elif p0.y > h:
        p0 = ln.intersect(bottom_edge)

    if p1.y < 0:
        p1 = ln.intersect(top_edge)
    elif p1.y > h:
        p1 = ln.intersect(bottom_edge)
    xs = [p0.x, p1.x]
    ys = [p0.y, p1.y]
    return xs, ys


def clip_line(xs, ys, w, h):
    """Clips a line to a rectangular area.
    from pysdl2 - cvutils

    cohensutherland(xmin, ymin, xmax, ymax,  x1, y1, x2, y2):
    This implements the Cohen-Sutherland line clipping algorithm.  xmin,
    ymax, xmax and ymin denote the clipping area, into which the line
    defined by x1, y1 (start point) and x2, y2 (end point) will be
    clipped.
    If the line does not intersect with the rectangular clipping area,
    four None values will be returned as tuple. Otherwise a tuple of the
    clipped line points will be returned in the form (cx1, cy1, cx2, cy2).
    """
    INSIDE,LEFT, RIGHT, LOWER, UPPER = 0, 1, 2, 4, 8

    #clip_line(x1, y1, x2, y2, xmax, ymax):
    #def fix_endpoints(xs, ys, w, h):
    xmin = 0.
    ymin = 0.
    xmax = float(w)
    ymax = float(h)
    x1, x2 = xs
    y1, y2 = ys

    def _getclip(xa, ya):
        #if dbglvl>1: print('point: '),; print(xa,ya)
        p = INSIDE  #default is inside

        # consider x
        if xa < xmin:
            p |= LEFT
        elif xa > xmax:
            p |= RIGHT

        # consider y
        if ya < ymin:
            p |= LOWER # bitwise OR
        elif ya > ymax:
            p |= UPPER #bitwise OR
        return p

    # check for trivially outside lines
    k1 = _getclip(x1, y1)
    k2 = _getclip(x2, y2)

    #%% examine non-trivially outside points
    #bitwise OR |
    while (k1 | k2) != 0: # if both points are inside box (0000) , ACCEPT trivial whole line in box

        # if line trivially outside window, REJECT
        if (k1 & k2) != 0: #bitwise AND &
            #if dbglvl>1: print('  REJECT trivially outside box')
            #return nan, nan, nan, nan
            return None

        #non-trivial case, at least one point outside window
        # this is not a bitwise or, it's the word "or"
        opt = k1 or k2 # take first non-zero point, short circuit logic
        if opt & UPPER:
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y = ymax
        elif opt & LOWER:
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y = ymin
        elif opt & RIGHT:
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x = xmax
        elif opt & LEFT:
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x = xmin
        else:
            raise RuntimeError('Undefined clipping state')

        if opt == k1:
            x1, y1 = x, y
            k1 = _getclip(x1, y1)
            #if dbglvl>1: print('checking k1: ' + str(x) + ',' + str(y) + '    ' + str(k1))
        elif opt == k2:
            #if dbglvl>1: print('checking k2: ' + str(x) + ',' + str(y) + '    ' + str(k2))
            x2, y2 = x, y
            k2 = _getclip(x2, y2)
    return [x1, x2], [y1, y2]


def fix_endpoints(xs, ys, w, h):
    """
    if endpoints are outside image,
    crop them to the image.
    admittedly hackish.
    """
    xs = map(float, xs)
    ys = map(float, ys)
    p0 = Point2(xs[0], ys[0])
    p1 = Point2(xs[1], ys[1])
    ln = Line2(p0, p1)

    top_left     = Point2(0., 0.)
    top_right    = Point2(w , 0.)
    bottom_left  = Point2(0., h)
    bottom_right = Point2(w , h)

    left_edge = Line2(top_left, bottom_left)
    right_edge = Line2(top_right, bottom_right)

    # first assume it intersects horizontal
    # edges, then clip
    # TODO edge case of vertical line

    p0 = ln.intersect(left_edge)
    p1 = ln.intersect(right_edge)

    cxscys = clip_line([p0.x, p1.x], [p0.y, p1.y], w, h)
    return cxscys


def abc_to_rhotheta(a, b, c):
    """
    TODO handle vertical lines
    ax + by + c = 0
    a = sin(theta)
    b = cos(theta)
    c = r
    theta in [-pi/2, pi/2)
    """
    theta = np.arctan(a/b)
    return (c, theta)


def rhotheta_to_endpoints(rho, theta, w, h):
    a = np.sin(theta)
    b = np.cos(theta)
    c = rho
    return abc_to_endpoints(a, b, c, w, h)


def endpoints_to_rhotheta(xs, ys):
    a, b, c = endpoints_to_abc(xs, ys)
    return abc_to_rhotheta(a, b, c)


def endpoints_to_center_rhotheta(xs, ys, w, h):
    a, b, c = endpoints_to_abc(xs, ys)
    return abc_to_center_rhotheta(a, b, c, w, h)


def endpoints_to_center_abc(xs, ys, w, h):
    a, b, c = endpoints_to_abc(xs, ys)
    return abc_to_center_abc(a, b, c, w, h)


def rhotheta_to_center_endpoints(rho, theta, w, h):
    a = np.sin(theta)
    b = np.cos(theta)
    c = rho
    a2, b2, c2 = abc_to_center_abc(a, b, c, w, h)
    return abc_to_center_endpoints(a2, b2, c2, w, h)


def abc_to_center_abc(a, b, c, w, h):
    """
    shift reference to center
    """
    x0 = w/2.
    y0 = h/2.
    return (a, b, a*x0 + b*y0 + c)


def abc_to_center_rhotheta(a, b, c, w, h):
    a2, b2, c2 = abc_to_center_abc(a, b, c, w, h)
    return abc_to_rhotheta(a2, b2, c2)


def abc_to_center_endpoints(a, b, c, w, h):
    raise Exception('not implemented')
