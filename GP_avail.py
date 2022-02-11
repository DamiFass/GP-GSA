import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
from scipy.io import loadmat 
import array
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic as RQ, ExpSineSquared as ESQ, DotProduct as DP, WhiteKernel as WK
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import make_scorer

from SALib.sample import saltelli 
from SALib.analyze import sobol 


def GridSearch(X,Y):

	y = np.atleast_2d(Y).T

	nPar = X.shape[1]
		
	kernel = C() * RBF()

	# # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
	# gps = GridSearchCV( GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40, normalize_y=True), param_grid={ "kernel": [ 
	# 											C(constant_value=10) + Matern(length_scale=nPar*[0.1], nu=2.5),  
	#  											C(constant_value=10) + Matern(length_scale=nPar*[1], nu=2.5), 
	#  											#C(constant_value=10) + Matern(length_scale=nPar*[10], nu=2.5), 
	#  											#C(constant_value=10) + Matern(length_scale=nPar*[100], nu=2.5),
	#  											C(constant_value=10) + Matern(length_scale=nPar*[0.1], nu=1.5),  
	#  											C(constant_value=10) + Matern(length_scale=nPar*[1], nu=1.5), 
	# # 											C(constant_value=10) + Matern(length_scale=nPar*[10], nu=1.5), 
	# # 											C(constant_value=10) + Matern(length_scale=nPar*[100], nu=1.5),
	# # 											C(constant_value=0.1) + Matern(length_scale=nPar*[1], nu=2.5),  
	# # 											C(constant_value=1) + Matern(length_scale=nPar*[1], nu=2.5), 
	# # 											C(constant_value=10) + Matern(length_scale=nPar*[1], nu=2.5), 
	#  											C(constant_value=100) + Matern(length_scale=nPar*[1], nu=2.5), ] }, 
	#  											cv=5, scoring=make_scorer(r2_score) )

	# # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
	gps = GridSearchCV( GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40, normalize_y=True), param_grid={ "kernel": [ 
												C(constant_value=10) + Matern(length_scale=nPar*[1], nu=2.5),  
	 											C(constant_value=10) + RBF(length_scale=nPar*[1]),]}, 
	 											cv=5, scoring=make_scorer(r2_score) )

	X_current = X[200:600,:]
	y_current = y[200:600,:]

	test_size = 0.2
	X_train, X_test, y_train, y_test = train_test_split(X_current, y_current, test_size=test_size, random_state=None)

	# print(GaussianProcessRegressor().get_params().keys())

	print(X_train.shape)
	print(y_train.shape)

	gps.fit(X_train, y_train)

	pd.set_option('display.max_columns', None)
	print( pd.DataFrame(gps.cv_results_) )

	print('#################################################################################################')
	# y_pred = gps.predict(X_train)

	# y_pred2 = gps.predict(X_test)

	# for i in range(y_test.shape[0]):
	# 	print( y_test[i], np.round(y_pred2[i],1) )

	print(gps.best_estimator_)
	print('#################################################################################################')
	print(gps.best_score_)
	print('#################################################################################################')
	print(gps.best_params_)
	print('#################################################################################################')
	print(gps.best_index_)
	print('#################################################################################################')
	print(gps.scorer_)
	print('#################################################################################################')

	y_pred = gps.predict(X_train)

	gp_best = gps.best_estimator_

	y_pred2, sigma2 = gp_best.predict(X_test, return_std=True)

	er1 = np.zeros((X_train.shape[0]))
	for i in range(X_train.shape[0]):
		er1[i] = abs( y_train[i] - y_pred[i] ) / max( y_train[i], y_pred[i] ) *100

	er2 = np.zeros((X_test.shape[0]))
	for i in range(X_test.shape[0]):
		er2[i] = abs( y_test[i] - y_pred2[i] ) / max( y_test[i], y_pred2[i] ) *100

	print('TRAINING SET:')
	print('Mean ABS err: ' + str( np.round( np.mean( abs( y_train - y_pred ) ),1 ) ) )
	print('Mean REL err: ' + str( np.round( np.mean( er1 ),1 ) ) )
	print('\n')
	print('#################################################################################################')
	print('\n')
	print('TEST SET:')
	print('Mean ABS err: ' + str( np.round( np.mean( abs( y_test - y_pred2 ) ),1 ) ) )
	print('Mean REL err: ' + str( np.round( np.mean( er2 ),1 ) ) )


	b = np.zeros(3)
	for j in range(len(er2)):
		if er2[j] <= 10:
			b[0] = b[0] + 1
		if er2[j] >10 and er2[j] <= 20:
			b[1] = b[1] + 1
		if er2[j] >20 and er2[j]:
			b[2] = b[2] + 1

	bb = b / len(er2)


	# fig = plt.figure()
	# ax = plt.axes(projection="3d")
	# num_bars = 3
	# x_pos = [1, 1.5, 2]
	# y_pos = [0] * num_bars
	# z_pos = [0] * num_bars
	# x_size = np.ones(num_bars) * 0.2
	# y_size = np.ones(num_bars) * 0.2
	# z_size = bb
	# rectNP = ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color='#035afc')
	# ax.set_xlim(0.5,2.5)
	# ax.set_ylim(0,2)
	# ax.set_xticks([1, 1.5, 2])
	# ax.set_yticks([])
	# ax.set_zticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
	# ax.set_zticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'], fontsize=8)
	# ax.set_xticklabels(['< 10%','10 < x < 20%','> 30%'], fontsize=8)
	# # ax2.set_yticklabels(ylabels, fontsize=16)
	# # ax2.legend(title='AP and Cai biomarkers comparison:', title_fontsize=14) 
	# ax.set_xlabel('Relative error', fontsize=12)
	# ax.set_zlabel('% of test set points', fontsize=12)
	# plt.show()

	print(gps.best_estimator_.kernel_)
	print('\n')
	print('LOG LIKELIHOOD: ' + str(gp_best.log_marginal_likelihood_value_) )

	# assex = np.linspace(0,y_test.shape[0] - 1 ,y_test.shape[0])
	# fig, ax = plt.subplots()
	# ax.plot(assex, y_test, 'r.', markersize=15, label='Data')
	# # ax.plot(assex, y_test_norm, 'r.', markersize=15, label='Data')
	# ax.scatter(assex, y_pred2, color='b', marker='o', label='Predictions')
	# ax.errorbar(assex, y_pred2, 1.96*sigma2, color='b', fmt='none', label='2*std')
	# ax.set_ylabel('Activation Time [ms]', fontsize=14)
	# ax.set_xticklabels([])
	# ax.legend()
	# plt.show()

	return 


def Emulator(X_whole,Y_whole,saveFlag,etichetta,types):

	in_dim = X_whole.shape[1]
	if np.ndim(Y_whole) == 1:
		Y_whole = np.atleast_2d(Y_whole).T
	num_features = Y_whole.shape[1]

	test_size = 0.2 
	X, X_test, Y, Y_test = train_test_split(X_whole, Y_whole, test_size=test_size, random_state=None)

	# 1st GridSearch: choose between polinomials
	pipe = Pipeline( [ ('poly', PolynomialFeatures()),('lr', LinearRegression(n_jobs=-1)) ] )

	# param_grid1 = {'poly__degree': [1,2,3]} 
	param_grid1 = {'poly__degree': [1]} 

	gs1 = GridSearchCV(pipe, param_grid1, n_jobs=-1, iid=False, cv=5, return_train_score=False)

	gs1.fit(X, Y)

	best_poly = gs1.best_estimator_

	residuals = Y - best_poly.predict(X)

	#ker = C(constant_value=10)*Matern(length_scale=in_dim*[0.1], nu=1.5)

	#emul = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=40)

	#print(cross_val_score(emul,X,residuals,cv=5))


	# 2nd GridSearch: choose between Kernels
	# param_grid2 = {'kernel': [C(constant_value=10)*Matern(length_scale=in_dim*[0.1], nu=i) for i in [1.5, 2.5]] + [C(constant_value=10)*RBF(length_scale=in_dim*[1.0])] + [C() + C(constant_value=10)*Matern(length_scale=in_dim*[0.1], nu=i) for i in [1.5, 2.5]] }

	param_grid2 = {'kernel': [C(constant_value=20)*Matern(length_scale=in_dim*[1], nu=1.5)] }

	#param_grid2 = {'kernel': [C(constant_value=10)*Matern(length_scale=in_dim*[0.1], nu=1.5)] + [C(constant_value=10)*Matern(length_scale=in_dim*[0.1], nu=2.5)] + [C(constant_value=10) + Matern(length_scale=in_dim*[1], nu=1.5)] + [C(constant_value=10) + Matern(length_scale=in_dim*[10], nu=1.5)] + [C(constant_value=10) + Matern(length_scale=in_dim*[100], nu=1.5)] + [C() + C()*RBF(length_scale=in_dim*[1.0])] + [C()*RBF(length_scale=in_dim*[0.1])] + [C()*RBF(length_scale=in_dim*[10])] }
	# param_grid2 = {'kernel': [C() + C()*RBF(length_scale=in_dim*[1.0]) + WK(noise_level=0.1)] + [C()*RBF(length_scale=in_dim*[1.0]) + WK(noise_level=1)] + [C()*RBF(length_scale=in_dim*[0.1])+ WK(noise_level=5)] + [C() +C()*RBF(length_scale=in_dim*[100])+ WK(noise_level=10)] + [C()*RBF(length_scale=in_dim*[1000])+ WK(noise_level=15)] }

	gs2 = GridSearchCV(GaussianProcessRegressor(n_restarts_optimizer=70, normalize_y=True), param_grid2, n_jobs=-1, iid=False, cv=5, return_train_score=False)

	gs2.fit(X, residuals)

	gp = gs2.best_estimator_

	print(gs2.best_estimator_)

	if saveFlag == 'y':
		name = 'GP_' + str(etichetta) + '_' + str(types)
		with open(name + '.pickle', 'wb') as f:
			pickle.dump(gp, f, protocol=pickle.HIGHEST_PROTOCOL)

		name = 'poly_' + str(etichetta) + '_' + str(types)
		with open(name + '.pickle', 'wb') as f:
			pickle.dump(best_poly, f, protocol=pickle.HIGHEST_PROTOCOL)

	# Test emulator:

	res_pred, sigma = gp.predict(X_test, return_std=True)

	y_pred = best_poly.predict(X_test) + res_pred 


	# Check performace: 
	y_pred = np.round(y_pred,1)
	yTrattino = np.mean(Y_test)
	somma = 0 
	somma2 = 0
	for i in range(Y_test.shape[0]):
		somma = somma + np.power( (y_pred[i]-Y_test[i]), 2 )	
		somma2 = somma2 + np.power( (Y_test[i]-yTrattino) , 2 )

	somma = 0
	err_rel = np.zeros(len(Y_test))
	errMedio = 0 

	for i in range(len(Y_test)):
		err = abs(y_pred[i] - Y_test[i])
		# err = abs(y_pred2[i] - y_test_norm[i])
		somma = somma + err
		massimo = max(y_pred[i], Y_test[i])
		err_rel[i] = err / massimo[0] * 100

	errMedio = somma/len(Y_test)
	err_rel_medio = np.mean(err_rel)

	r2 = r2_score(Y_test, y_pred)
	
	print('r2 score {} = {}'.format(etichetta,r2))

	if saveFlag == 'y':
		np.save('r2Score_{}'.format(etichetta),r2)

	print('Mean Abs Err = ' + str(errMedio) )
	# print(err_rel)
	print('Mean err_rel_medio = ' + str(round(err_rel_medio,1)) )


	b = np.zeros(3)

	for j in range(len(err_rel)):
		if err_rel[j] <= 10:
			b[0] = b[0] + 1
		if err_rel[j] >10 and err_rel[j] <= 20:
			b[1] = b[1] + 1
		if err_rel[j] >20 and err_rel[j]:
			b[2] = b[2] + 1

	bb = b / len(err_rel)
	print('Percentage of points with relative error < 10% : ' + str( round(bb[0]*100) ) )
	print('Percentage of points with 10% > relative error < 20% : ' + str( round(bb[1]*100) ) )
	print('Percentage of points with relative error > 20% : ' + str( round(bb[2]*100) ) )

	print('\n')
	print(y_pred[:30])

	#fig = plt.figure()
	#ax = plt.axes(projection="3d")
	#num_bars = 3
	#x_pos = [1, 1.5, 2]
	#y_pos = [0] * num_bars
	#z_pos = [0] * num_bars
	#x_size = np.ones(num_bars) * 0.2
	#y_size = np.ones(num_bars) * 0.2
	#z_size = bb * 100
	#rectNP = ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color='#035afc')
	#ax.set_xlim(0.5,2.5)
	#ax.set_ylim(0,2)
	#ax.set_xticks([1, 1.5, 2])
	#ax.set_yticks([])
	#ax.set_zticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	#ax.set_zticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], fontsize=8)
	#ax.set_xticklabels(['< 10%','10 < x < 20%','> 20%'], fontsize=8)
	# ax2.set_yticklabels(ylabels, fontsize=16)
	# ax2.legend(title='AP and Cai biomarkers comparison:', title_fontsize=14) 
	# ax.set_xlabel('Relative error', fontsize=12)
	#ax.set_zlabel('% of test set points', fontsize=12)
	#ax.set_title('Relative error', fontsize=16)
	#plt.show()

	Y_test_ord = np.sort(Y_test)
	indici = []
	for i in Y_test_ord:
		indici.append( np.where(( Y_test==i ))[0][0] )

	assex = np.linspace(0,Y_test_ord.shape[0] - 1 ,Y_test_ord.shape[0])
	fig, ax = plt.subplots()
	ax.plot(assex, Y_test_ord, 'r.', markersize=15, label='Data')
	# ax.plot(assex, y_test_norm, 'r.', markersize=15, label='Data')
	ax.scatter(assex, y_pred[indici], color='b', marker='o', label='Predictions')
	ax.errorbar(assex, y_pred[indici], 1.96*sigma, color='b', fmt='none', label='2*std')
	ax.set_ylabel('Activation Time [ms]', fontsize=14)
	ax.set_xticklabels([])
	ax.legend()
	plt.show()


	# assex = np.linspace(0,Y_test.shape[0] - 1 ,Y_test.shape[0])
	# fig, ax = plt.subplots()
	# ax.plot(assex, Y_test, 'r.', markersize=15, label='Data')
	# # ax.plot(assex, y_test_norm, 'r.', markersize=15, label='Data')
	# ax.scatter(assex, y_pred, color='b', marker='o', label='Predictions')
	# ax.errorbar(assex, y_pred, 1.96*sigma, color='b', fmt='none', label='2*std')
	# ax.set_ylabel('Activation Time [ms]', fontsize=14)
	# ax.set_xticklabels([])
	# ax.legend()
	# plt.show()

	return gp, best_poly



def GSA(gp,poly,ranges,labels,saveFlag,etichetta,types):

	nPar = ranges.shape[1]

	problem = {'num_vars': nPar, 'names': labels, 'bounds': ranges.transpose() }

	# Generate samples
	param_values = saltelli.sample(problem, 4000)

	# Run gp --> predict the residuals! 
	res_pred, sigma = gp.predict(param_values, return_std=True)

	# Run polynomial regressor AND add the residuals --> predict the whole output
	y_pred = poly.predict(param_values) + res_pred

	# Perform analysis
	Si = sobol.analyze(problem, np.squeeze(y_pred), print_to_console=False)



	# # print(Si['S1'])

	# # Dictionary with Sensitivity Index to Pandas Data Frame:
	M = Si
	keys = ['S1', 'S1_conf', 'ST', 'ST_conf', 'S2', 'S2_conf']
	names = labels
	
	v = []
	v_conf = []
	nn = []
	for i in range(6):
		for j in range(6):
	 		if j > i:
	 			if M['S2'][i, j] < 0.01:
	 				v.append( 0.0 )
	 				v_conf.append( 0.0 )
	 			else:
	 				v.append( M['S2'][i, j] )
	 				v_conf.append( M['S2_conf'][i, j] )
	 			nn.append( (names[i], names[j]) )
	M2 = np.hstack((np.array(v).reshape(-1, 1), np.array(v_conf).reshape(-1, 1)))
	
	l1 = np.where(M['S1'] < 0.01)[0]
	M['S1'][l1] = np.zeros((len(l1),), dtype=float)
	M['S1_conf'][l1] = np.zeros((len(l1),), dtype=float)
	d1 = {keys[i]: M[keys[i]] for i in range(2)}
	first_Si = pd.DataFrame(data=d1, index=names)

	lt = np.where(M['ST'] < 0.01)[0]
	M['ST'][lt] = np.zeros((len(lt),), dtype=float)
	M['ST_conf'][lt] = np.zeros((len(lt),), dtype=float)
	dt = {keys[i]: M[keys[i]] for i in range(2, 4)}
	total_Si = pd.DataFrame(data=dt, index=names)
	
	d2 = {keys[i]: M2[:, i-4] for i in range(4, 6)}
	second_Si = pd.DataFrame(data=d2, index=nn)
	
	if saveFlag == 'y':
		np.save('S1' + etichetta + '_' + types,d1['S1'])
		np.save('S1_conf' + etichetta + '_' + types,d1['S1_conf'])
		np.save('ST' + etichetta + '_' + types,dt['ST'])
		np.save('ST_conf' + etichetta + '_' + types,dt['ST_conf'])
	

	#print(d1['S1'])
	#print(dt['ST'])

	#store = pd.HDFStore('store.h5')
	#store['firstOrder' + types[ww] ] = first_Si  # save it
	#store['TotalEffect' + types[ww] ] = total_Si  # save it


	# # Plot first order index:
	# CONF_COLUMN = '_conf'
	# conf_cols = first_Si.columns.str.contains(CONF_COLUMN)
	# confs = first_Si.loc[:, conf_cols]
	# confs.columns = [c.replace(CONF_COLUMN, '') for c in confs.columns]
	# Sis = first_Si.loc[:, ~conf_cols]
	# ax = Sis.plot(kind='bar', yerr=confs)
	# plt.show(ax)

	# # Plot total effect:
	# CONF_COLUMN = '_conf'
	# conf_cols = total_Si.columns.str.contains(CONF_COLUMN)
	# confs = total_Si.loc[:, conf_cols]
	# confs.columns = [c.replace(CONF_COLUMN, '') for c in confs.columns]
	# Sis = total_Si.loc[:, ~conf_cols]
	# ax = Sis.plot(kind='bar', yerr=confs)
	# plt.show(ax)

	return 

