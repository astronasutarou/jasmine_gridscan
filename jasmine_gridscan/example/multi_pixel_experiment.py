#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..pixel import Pixel, gaussian_response
from ..grid import Grid
from ..experiment import Experiment, PixelArray


if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap('multi_pixel_experiment',
    description='''
    Multiple pixel experiment of grid scan modulation. This function
    simulates the outputs of multiple pixels in the grid scan experiment.
    The locations of the pixels relative to Pixel[1,1] is estimated from
    the phases of the modulations.''')

  parser.add_argument(
    '-N', '--size', type=int, default=6, metavar='N',
    help='the size of the detector (pixel).')
  parser.add_argument(
    '-S', '--scale', type=float, default=10.0, metavar='d',
    help='the typical pixel-to-pixel separation (um).')
  parser.add_argument(
    '-s', '--sigma', type=float, default=0.5, metavar='sig',
    help='the scale of the pixel position scatter (um).')
  parser.add_argument(
    '--seed', type=int, default=42, metavar='seed',
    help='the seed value of the random generator')
  parser.add_argument(
    '-n', '--num', type=int, default=1100, metavar='N',
    help='the number of measurements.')
  parser.add_argument(
    '--tstart', type=float, default=100, metavar='t',
    help='the starting time.')
  parser.add_argument(
    '--tend', type=float, default=300, metavar='t',
    help='the end time.')
  parser.add_argument(
    '--transp_width', type=float, default=6.0, metavar='w',
    help='the width of the transparent parts of the grid (um).')
  parser.add_argument(
    '--opaque_width', type=float, default=15.0, metavar='w',
    help='the width of the opaque parts of the grid (um).')
  parser.add_argument(
    '--slit_number', type=int, default=100, metavar='N',
    help='the total number of the slits in the grid.')
  parser.add_argument(
    '--velocity', type=float, default=1.0, metavar='v',
    help='the velocity of grid scanning (um/s).')

  args = parser.parse_args()

  time = np.linspace(args.tstart,args.tend,args.num)
  grid = Grid(
    transparent_width=args.transp_width,
    opaque_width=args.opaque_width,
    slit_number=args.slit_number,
    velocity=args.velocity)
  parr = PixelArray(size=args.size,
    scale=args.scale, scatter=args.sigma, seed=args.seed)
  exp = Experiment(parr, grid, time)

  position = exp.estimate_position()

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot()
  ax.scatter(parr.x-parr.x[0], parr.y-parr.y[0],
             label='input position')
  ax.scatter(position[:,1], position[:,0], marker='+',
             label='estimated position')
  ax.set_xlabel('relative position from pixel[1,1] ($\mu$m)', fontsize=14)
  ax.set_ylabel('relative position from pixel[1,1] ($\mu$m)', fontsize=14)
  ax.legend(bbox_to_anchor=(1,1), loc='lower right', frameon=False, ncol=2)
  fig.tight_layout()
  plt.show()
