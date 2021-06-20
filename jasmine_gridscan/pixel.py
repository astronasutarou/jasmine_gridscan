#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from functools import reduce
from operator import add
from dataclasses import dataclass
from typing import Callable
from scipy.special import gamma, gammainc


def gaussian_response(x, width=8, order=8):
  temp = -np.sign(x)*(1-gammainc(1.0/order,(x/(width/2))**order))
  return (temp+np.sign(x)+1)/2


@dataclass(frozen=True)
class Pixel(object):
  ''' Individual Pixel class

  Attributes:
    x       : x-coordinate of the pixel center (um).
    y       : y-coordinate of the pixel center (um).
    response: integrated one-dimensional pixel response function.
  '''
  x:        float
  y:        float
  response: Callable

  def evaluate(self, axis, t, grid):
    slits = grid.abscissa_list()
    s,e = np.expand_dims(slits[:,0],1), np.expand_dims(slits[:,1],1)
    x0 = np.expand_dims(np.array(t)*grid.velocity,0)
    xs,xe = x0+s-getattr(self, axis), x0+e-getattr(self, axis)
    return np.sum(self.response(xe)-self.response(xs),axis=0)

  def evaluate_x(self, t, grid):
    return self.evaluate('x', t, grid)

  def evaluate_y(self, t, grid):
    return self.evaluate('y', t, grid)
