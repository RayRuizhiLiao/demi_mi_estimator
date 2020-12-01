import numpy as np
from math import log


def gaussian_generator(mean, cov, num_samples):
	""" A generator that generates samples from 
	two n-dimensional Gaussian distributions x and y
	and their joint distributon (2n-dimensional).
	
	Args:
		mean: a 2n-dimensinoal vector, the mean of
			the joint Gaussian distribution ([m_x, m_y]).
		cov: a 2n-by-2n covariance matrix of the joint Gaussian distribution
			([sigma_xx, sigma_xy; sigma_yx, sigma yy]).
		num_samples: the number of samples to be drawn from 
			the joint Gaussian distribution.

	Returns:
		samples_x: samples drawn from x (num_samples-by-n).
		samples_y: samples drawn from y (num_samples-by-n).
		samples_xy: samples drawn from xy (num_samples-by-2n).
	"""
	assert len(mean)%2 == 0, \
		"The mean vector should be a concatenation of two euqally long vectors"
	assert np.shape(cov)[0]==np.shape(cov)[1] and np.shape(cov)[0]==len(mean), \
		"The covariance matrix size should be 2n-by-2n, "\
		"where 2n is the length of the mean vector"

	n = int(len(mean)/2)
	mean_x = mean[:n]
	mean_y = mean[n:]

	cov_xx = cov[:n, :n]
	cov_yy = cov[n:, n:]

	return np.random.multivariate_normal(mean_x, cov_xx, num_samples), \
		np.random.multivariate_normal(mean_y, cov_yy, num_samples), \
		np.random.multivariate_normal(mean, cov, num_samples)

def cov_to_mi(cov):
	n = int(len(cov)/2)
	cov_xx = cov[:n, :n]
	cov_yy = cov[n:, n:]

	det_xx = np.linalg.det(cov_xx)
	det_yy = np.linalg.det(cov_yy)
	det = np.linalg.det(cov)

	return 0.5*log(det_xx*det_yy/det)

def mi_to_rho(gaussian_length, mi):
	"""Obtain the rho for Gaussian give ground truth mutual information."""
	return np.sqrt(1 - np.exp(-2.0 / gaussian_length * mi))

def generate_gaussian_samples(gaussian_length, rho, num_samples):
	mean = np.zeros([2*gaussian_length])

	assert rho<1, \
		"The diagnoal covariance value should be smaller than 1!"

	cov = np.zeros([2*gaussian_length, 2*gaussian_length])
	for i in range(2*gaussian_length):
		row = np.zeros([2*gaussian_length])
		row[i] = 1
		if i < gaussian_length:
			row[i+gaussian_length] = rho
		else:
			row[i-gaussian_length] = rho
		cov[i] = row

	x, y, xy = gaussian_generator(mean, cov, num_samples)

	return x, y, xy, cov_to_mi(cov)