#It is recommended that the functions that are built receive data of type np.array
import numpy as np
import scipy.stats as stats

def moment(data, n):
	'''Calculate the moment n of a distribution data'''
	return stats.moment(data, moment=n)

def quantile(data, c):
	'''Calculate the quantile of a distribution data'''
	return np.quantile(data, c)

def threshold_up(data, threshold):
	'''Calculates the average number of data over the threshold'''
	return (threshold < data).mean()

def threshold_down(data, threshold):
	'''Calculates the average number of data under the threshold'''
	return (threshold > data).mean()

def fourier_trans(f):
	'''Calculate the fourier transformation of f '''
	pass

if __name__ == '__main__':
	
	#MOMENTS_TEST. 
	test = np.array([1, 2, 3, 4, 5])
	print(moment(test, 2.0))
	print('---------------')
	
	#QUANTILES_TEST
	arr = np.array([20, 2, 7, 1, 34]) 
	print(quantile(arr, .50))
	print('---------------')
	
	#THRESHOLD_TEST
	a = np.array([ 1,  2, 11, 13,  5, 15,  7,  8,  9, 10])
	threshold = 10
	print(threshold_up(a, threshold))
	print(threshold_down(a, threshold))
