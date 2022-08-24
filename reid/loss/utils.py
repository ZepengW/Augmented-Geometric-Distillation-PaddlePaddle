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
    m, n = x.shape[0], y.shape[0]
    #xx = paddle.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    xx = paddle.expand(paddle.pow(x, 2).sum(1, keepdim=True), shape=[m, n])
    #yy = paddle.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    yy = paddle.transpose(paddle.expand(paddle.pow(y, 2).sum(1, keepdim=True), shape=[n, m]), [1,0])
    dist = xx + yy
    #dist.addmm_(1, -2, x, paddle.transpose(y, [1,0]))
    dist = paddle.addmm(dist, x, paddle.transpose(y, [1,0]), -2, 1)
    
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist = paddle.sqrt(paddle.clip(dist, min=1e-12))
    return dist