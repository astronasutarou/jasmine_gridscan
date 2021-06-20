#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from ..pixel import gaussian_response


if __name__ == '__main__':
  from argparse import ArgumentParser as ap
  parser = ap('illustrate_pixel_response')

  parser.add_argument(
    '-N', '--size', type=int, default=100,
    help='the image size.')
  parser.add_argument(
    '-w', '--width', type=float, default=8,
    help='the width of the response function.')
  parser.add_argument(
    '--order', type=int, default=8,
    help='the order of the response function')

  args = parser.parse_args()

  x = np.linspace(-args.width,args.width,args.size)
  xp = np.linspace(-args.width,args.width,args.size+1)
  response = np.diff(gaussian_response(xp,width=args.width,order=args.order))
  response = np.expand_dims(response, axis=1)/np.diff(xp).mean()
  resimg = response*response.T

  import matplotlib.gridspec as gp
  fig = plt.figure(figsize=(8,10), constrained_layout=False)
  gs = fig.add_gridspec(4,1)
  ax1 = fig.add_subplot(gs[0,0])
  ax1.plot(x, response)
  ax1.set_xlabel('relative position ($\mu$m)', fontsize=14)
  ax1.set_ylabel('norm. response', fontsize=14)
  ax2 = fig.add_subplot(gs[1:,0], sharex=ax1)
  ax2.imshow(resimg,extent=[x.min(),x.max(),x.min(),x.max()],origin='lower')
  ax2.set_xlabel('relative position ($\mu$m)', fontsize=14)
  ax2.set_ylabel('relative position ($\mu$m)', fontsize=14)
  fig.tight_layout()
  plt.show()
