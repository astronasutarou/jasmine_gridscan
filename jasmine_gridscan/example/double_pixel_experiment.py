#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..pixel import Pixel, gaussian_response
from ..grid import Grid
from ..experiment import calc_phase


if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap('double_pixel_experiment',
    description='''
    A two pixel experiment of the grid scan modulation. This function
    simulates the outputs of two pixels in the grid scan experiment.
    The displacement of the two pixels is esimated using the phase
    difference between the two light curves.''')

  parser.add_argument(
    '-d', '--delta', type=float, default=3.0, metavar='d',
    help='the distance between two pixels (um).')
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
  pix1 = Pixel(1.0, 1.0, gaussian_response)
  pix2 = Pixel(1.0+args.delta, 1.0, gaussian_response)
  cnt1 = pix1.evaluate_x(time, grid)
  cnt2 = pix2.evaluate_x(time, grid)

  ph1,fit1 = calc_phase(time, cnt1, grid, with_fit=True)
  ph2,fit2 = calc_phase(time, cnt2, grid, with_fit=True)

  C = grid.slit_interval/np.pi
  d = C*(ph2-ph1)
  print(f'input delta    : {args.delta:.3f} um.')
  print(f'estimated delta: {d:.3f} um.')

  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot()
  ax.plot(time, cnt1, label='Pixel[1.0,1.0]')
  ax.plot(time, fit1, '--', alpha=0.5)
  ax.plot(time, cnt2, label=f'Pixel[{1+args.delta:.1f},1.0]')
  ax.plot(time, fit2, '--', alpha=0.5)
  ax.set_xlabel('time (s)',fontsize=14)
  ax.set_ylabel('pixel count (ADU)',fontsize=14)
  ax.set_ylim([-0.05,0.95])
  ax.legend(loc='upper right', frameon=False, fontsize=14, ncol=2)
  fig.tight_layout()
  plt.show()
