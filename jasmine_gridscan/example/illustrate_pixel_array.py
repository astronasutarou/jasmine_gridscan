#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..pixel import gaussian_response
from ..experiment import PixelArray


if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap('illustrate_pixel_response')

  parser.add_argument(
    '-N', '--size', type=int, default=6, metavar='N',
    help='the size of the detector (pixel).')
  parser.add_argument(
    '-w', '--width', type=float, default=8.0,
    help='the width of the response function.')
  parser.add_argument(
    '--order', type=int, default=8,
    help='the order of the response function')
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
    '-r', '--resolution', type=int, default=1024, metavar='N',
    help='the image resolution.')

  args = parser.parse_args()

  xmin,xmax = -1*args.width, (args.size-1)*args.scale+args.width
  x  = np.linspace(xmin, xmax, args.resolution)
  xp = np.linspace(xmin,xmax,args.resolution+1)
  image = np.zeros((args.resolution,args.resolution))

  resp = lambda X: gaussian_response(X,width=args.width,order=args.order)
  parr = PixelArray(size=args.size, scale=args.scale,
                    response=resp, scatter=args.sigma, seed=args.seed)
  for p in parr.pixels:
    resx = np.expand_dims(np.diff(resp(xp-p.x)),axis=1)
    resy = np.expand_dims(np.diff(resp(xp-p.y)),axis=1)
    image = image + (resx*resy.T).T

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot()
  ax.imshow(image, extent=[xmin,xmax,xmin,xmax], origin='lower')
  ax.scatter(parr.x, parr.y, marker='+')
  ax.set_xlabel('detector position ($\mu$m)', fontsize=14)
  ax.set_ylabel('detector position ($\mu$m)', fontsize=14)
  fig.tight_layout()
  plt.show()
