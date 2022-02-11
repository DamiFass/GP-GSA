from itertools import combinations
import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.special import binom
import seaborn as sns
import sys
import timeit
import pickle
from scipy.stats import multivariate_normal

start_time = timeit.default_timer()

def f_emul(gp,poly,X_new,n_samples,seed,svm,ranges,N): 
	X_whole = np.zeros((X_new.shape[0],X_new.shape[1]))
	for i in range(X_new.shape[1]):
		X_whole[:,i] = X_new[:,i]/ranges[i,1]

	if svm != 0:
		label = svm.predict(X_whole)

		conta1 = 0
		contameno1 = 0
		for i in range(label.shape[0]):
			if label[i] == 1:
				conta1 = conta1 + 1
			else:
				contameno1 = contameno1 + 1

		print('With {} input points (N), X Sobol has dimensions: {}'.format(N,X_new.shape))		
		print('SVM makes {} predictions, which should be equal to {}, the number of X Sobol rows'.format(label.shape[0],X_whole.shape[0]))
		print('SVM predicts {} points to be discarded over {} points, which is {} per cent'.format(contameno1,label.shape[0],np.round(contameno1/label.shape[0]*100,1) ) )
		# print('conta1 {}'.format(conta1))
		# print('contameno1 {}'.format(contameno1))

		lista = np.where(label==-1)[0]

	res, cov = gp.predict(X_new, return_cov=True)
	if svm != 0:
		res[lista,:] = np.zeros((len(lista),1)) 

	mean = poly.predict(X_new)
	if svm != 0:
		mean[lista,:] = np.zeros((len(lista),1))

	y_sample = multivariate_normal.rvs( np.ndarray.flatten(res+mean) , cov, size=n_samples, random_state=seed).T

	# if etic == 'SIsvm':
	# 	for jj in lista:
	# 		y_sample[jj,:] = np.zeros((1,n_samples))

	return y_sample


SEED = 8

def main():
	seed = SEED
	random.seed(seed)
	np.random.seed(seed)

	# Usage:
	# python3 run_gsa_mio.py ranges, num_par, GP, poly, folder to save indexes, num points Sobol sequence, SVM.
	# THe script can be called with 5 or 6 parameters, depending whether or not there's the SVM!

	ranges = np.load(sys.argv[1])
	
	if int(sys.argv[2]) == 10:    # EPI_ENDO - ENDO_EPI
		labels = ['Scar radius','Scar depth','Scar conductivity','Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 10
	if int(sys.argv[2]) == 9:     # TRANSMURAL
		labels = ['Scar radius','Scar conductivity','Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 9
	if int(sys.argv[2]) == 8:     # BLOCKED
		labels = ['Scar radius','Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 8
	if int(sys.argv[2]) == 7:     # FIXED
		labels = ['Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 7
	
	GP_path = sys.argv[3]
	poly_path = sys.argv[4]

	tag = sys.argv[5]

	sobol_seq_points = int(sys.argv[6])

	if len(sys.argv) == 8:
		svm_path = sys.argv[7]
		with open(svm_path, 'rb') as f:
			svm = pickle.load(f)
	else:
		svm = 0

	#========================
	# GPE loading
	#========================

	with open(GP_path, 'rb') as f:
		gp = pickle.load(f)

	with open(poly_path, 'rb') as f:
		poly = pickle.load(f)
	
	#========================
	# SA LIB
	#========================
	N = sobol_seq_points # deafult 1000, try 2000, 3000 and 5000                             
	D = len(labels)
	n_draws = 1000

	I = ranges
	index_i = labels
	index_ij = ['({}, {})'.format(c[0], c[1]) for c in combinations(index_i, 2)]

	problem = {
		'num_vars': D,
		'names': index_i,
		'bounds': I
	}

	X_sobol = saltelli.sample(problem, N, calc_second_order=True) # N x (2D + 2)


	Y = f_emul(gp, poly, X_sobol, n_draws, seed, svm, I, N)


	ST = np.zeros((0, D), dtype=float)
	S1 = np.zeros((0, D), dtype=float)
	S2 = np.zeros((0, int(binom(D, 2))), dtype=float)
	for i in range(n_draws):
		S = sobol.analyze(problem, Y[:,i], calc_second_order=True, parallel=True, n_processors=24, seed=seed)
		total_order, first_order, (_, second_order) = sobol.Si_to_pandas_dict(S)
		ST = np.vstack((ST, total_order['ST'].reshape(1, -1)))
		S1 = np.vstack((S1, first_order['S1'].reshape(1, -1)))
		S2 = np.vstack((S2, np.array(second_order['S2']).reshape(1, -1)))

	print('GSA - Elapsed time: {:.1f} min'.format( (timeit.default_timer() - start_time)/60 ))

	np.savetxt(str(tag) + '/STi.txt', ST, fmt='%.6f')
	np.savetxt(str(tag) + '/Si.txt', S1, fmt='%.6f')
	np.savetxt(str(tag) + '/Sij.txt', S2, fmt='%.6f')

	df_ST = pd.DataFrame(data=ST, columns=index_i)
	df_S1 = pd.DataFrame(data=S1, columns=index_i)
	df_S2 = pd.DataFrame(data=S2, columns=index_ij)

	# gs = grsp.GridSpec(2, 2)
	# fig = plt.figure(figsize=(2*8.27, 4*11.69/3))
	# ax0 = fig.add_subplot(gs[0, 0])
	# ax1 = fig.add_subplot(gs[0, 1])
	# ax2 = fig.add_subplot(gs[1, :])
	# sns.boxplot(ax=ax0, data=df_S1)
	# sns.boxplot(ax=ax1, data=df_ST)
	# sns.boxplot(ax=ax2, data=df_S2)
	# ax0.set_ylim(0, 1)
	# ax0.set_title('First-order effect', fontweight='bold', fontsize=12)
	# ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, horizontalalignment='right')
	# ax1.set_ylim(0, 1)
	# ax1.set_title('Total effect', fontweight='bold', fontsize=12)
	# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
	# ax2.set_ylim(0, 1)
	# ax2.set_title('Second-order effect', fontweight='bold', fontsize=12)
	# ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
	# plt.savefig('si_distr_salib_{}_{}_2.png'.format(N, etichette[lol]))

if __name__ == '__main__':
	main()

