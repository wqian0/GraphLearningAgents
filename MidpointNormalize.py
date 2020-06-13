import numpy as np
from numpy import random as nrd
import random as rd
import copy
from copy import deepcopy
from random import choice
from array import *
import time
import os, glob
import small_world as sw
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import scipy.ndimage as sim
from mpl_toolkits.mplot3d import Axes3D
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))