#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
from typing import List, Callable
from dataclasses import dataclass, field

from .pixel import Pixel, gaussian_response
from .grid import Grid


def calc_phase(t, c, grid, with_fit=False):
  tic = np.pi/grid.slit_interval*grid.velocity*t
  _cos = lambda n,p: np.cos(n*(tic-np.pi*np.tanh(p)))
  func = lambda A: \
    A[0]+A[1]*_cos(1,A[4])+A[2]*_cos(2,A[4])+A[3]*_cos(2,A[4])
  x0 = np.array([np.mean(c),0,0,0,0])
  res = opt.minimize(lambda A: np.square(c-func(A)).sum(),x0)
  if with_fit is True:
    return np.pi*np.tanh(res.x[4]), func(res.x)
  else:
    return np.pi*np.tanh(res.x[4])


@dataclass
class PixelArray(object):
  '''
  '''
  size:     int         = 4
  scale:    float       = 10.0
  scatter:  float       = 1.0
  response: Callable    = gaussian_response
  pixels:   List[Pixel] = field(init=False)
  seed:     int         = 42

  def __post_init__(self):
    xarr = yarr = self.scale*np.arange(self.size)
    xx,yy = np.meshgrid(xarr,yarr)
    np.random.seed(self.seed)
    xx = xx + np.random.normal(0.0,self.scatter,size=(self.size,self.size))
    yy = yy + np.random.normal(0.0,self.scatter,size=(self.size,self.size))
    self.pixels = [Pixel(x,y,self.response) for x,y in zip(xx.flat,yy.flat)]


  @property
  def position(self):
    return np.array([np.array(_.y,_.x) for _ in self.pixels])
  @property
  def x(self):
    return np.array([_.x for _ in self.pixels])
  @property
  def y(self):
    return np.array([_.y for _ in self.pixels])


@dataclass(frozen=True)
class Experiment(object):
  '''
  '''
  pixel: PixelArray
  grid:  Grid
  time:  np.array

  def modulation(self, axis):
    counts = list()
    if axis == 'x':
      for p in tqdm(self.pixel.pixels):
        counts.append(p.evaluate_x(self.time, self.grid))
    elif axis == 'y':
      for p in tqdm(self.pixel.pixels):
        counts.append(p.evaluate_y(self.time, self.grid))
    else:
      raise ValueError(f'wrong axis "{axis}" specified ("x" or "y").')
    return np.array(counts)

  def estimate_phase(self, axis, counts=None, with_fit=False):
    if counts is None: counts = self.modulation(axis)
    phase = list()
    fit   = list()
    for count in tqdm(counts):
      p,f = calc_phase(self.time,count,self.grid,with_fit=True)
      phase.append(p)
      fit.append(f)
    if with_fit is True:
      return np.array(phase),np.array(fit)
    else:
      return np.array(phase)

  def estimate_position(self):
    ## "ref" is the reference poistion of the pixel.
    arr = np.round(
      2*self.pixel.scale/self.grid.slit_interval*np.arange(self.pixel.size))
    xx,yy = np.meshgrid(arr,arr)
    ref_x = self.grid.slit_interval*xx.flatten()/2
    ref_y = self.grid.slit_interval*yy.flatten()/2
    phase_x  = self.estimate_phase('x')
    phase_y  = self.estimate_phase('y')
    dx = ref_x+self.grid.slit_interval/np.pi*(phase_x-phase_x[0])
    dy = ref_y+self.grid.slit_interval/np.pi*(phase_y-phase_y[0])
    return np.array([np.array([y,x]) for y,x in zip(dy,dx)])
