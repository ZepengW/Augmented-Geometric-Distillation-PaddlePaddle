# -*- coding: utf-8 -*-
# Time    : 2021/8/6 16:37
# Author  : Yichen Lu
import paddle


def cos(x, y):
    x_ = x / x.norm(dim=1, keepdim=True)
    y_ = y / y.norm(dim=1, keepdim=True)
    return paddle.mm(x_, y_.t())


def euclidean_dist(x, y, *args):
    """
    Args:
      x: pypaddle Variable, with shape [m, d]
      y: pypaddle Variable, with shape [n, d]
    Returns:
      dist: pypaddle Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = paddle.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = paddle.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist