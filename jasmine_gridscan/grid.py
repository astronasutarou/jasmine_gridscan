#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Grid(object):
  ''' Grid class

  Attributes:
    transparent_width: the width of the transparent slit (um).
    opaque_width     : the width of th opaque grid line (um).
    slit_number      : the total number of the grid lines.
    velocity         : the grid scanning speed (um/s).
    slit_inberval    : the interval betwen transparent slits (um).
  '''
  transparent_width: float = 6.0
  opaque_width:      float = 15.0
  slit_number:       int   = 100
  velocity:          float = 1.0
  slit_interval:     float = field(init=False)

  def __post_init__(self):
    object.__setattr__(self, 'slit_interval',
        self.transparent_width+self.opaque_width)

  def abscissa(self, k: int):
    half = self.transparent_width/2.0
    if k<0:
      return np.array([half+self.opaque_width,1e8])
    if k>=self.slit_number:
      return np.array([-1e8,-self.slit_number*self.slit_interval+half])
    x0 = -k*self.slit_interval+half
    return np.array([x0-self.transparent_width, x0])

  def abscissa_list(self):
    ''' list of transparent region
    '''
    return np.array([self.abscissa(k) for k in range(-1,self.slit_number+1)])
