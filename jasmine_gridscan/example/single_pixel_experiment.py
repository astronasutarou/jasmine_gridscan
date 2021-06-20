#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..pixel import Pixel, gaussian_response
from ..grid import Grid


if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap('single_pixel_experiment',
    description='''
    A single pixel experiment of grid scan modulation. This function
    simulates the output of a single pixel in the grid scan experiment.''')

  parser.add_argument(
    '-n', '--num', type=int, default=1000, metavar='N',
    help='the number of measurements.')
  parser.add_argument(
    '--tstart', type=float, default=0, metavar='t',
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
  pix = Pixel(1.0, 1.0, gaussian_response)
  count = pix.evaluate_x(time, grid)

  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot()
  ax.plot(time, count, label='Pixel[1,1]')
  ax.set_xlabel('time (s)',fontsize=14)
  ax.set_ylabel('pixel count (ADU)',fontsize=14)
  ax.set_ylim([-0.05,0.95])
  ax.legend(loc='upper right', frameon=False, fontsize=14)
  fig.tight_layout()
  plt.show()
