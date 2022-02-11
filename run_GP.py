import numpy as np 
import pickle
import matplotlib.pyplot as plt 
from scipy.io import loadmat 
import array
import sys

import GP_avail as gpa 

#dataset = int(input('Choose data set: 1-INITIAL_EPI_ENDO     2-TRANSUMRAL     3-TOATL_BLOCK     4-TOTAL_FIXED'))

# dataset = np.int(sys.argv[1])
# print(dataset)

Y = np.loadtxt(str(sys.argv[1]))
X = np.loadtxt(str(sys.argv[2]))   # LH_GP, not LH_sim!!
xlabels = sys.argv[3]

if X.shape[1] == 10:
	labels = ['Scar radius','Scar depth','Scar conductivity','Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 10
if X.shape[1] == 9:
	labels = ['Scar radius','Scar conductivity','Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 9 
if X.shape[1] == 8:
	labels = ['Scar radius','Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 8
if X.shape[1] == 7:
	labels = ['Internal bath','EHT thickness','EHT conductivity','CP thickness','CP conductivity','EHT-tissue contact area','Delta thickness'] # 7

# Isolate first row of Y (i.e. the top right corner activation times) and eliminate the rows where AT = -1, both from Y and X 
cm1 = 0
Ylista = []
for i in range(Y.shape[0]):
	if Y[i] == -1:
		cm1 = cm1 + 1
	else:
		Ylista.append(Y[i])
#Y = np.array(Ylista)
#X = np.zeros((Y.shape[0], Xbig.shape[1]))
#conta = 0
#for i in range(Y):
#for i in range(999):
#	if Ybig[i,0] != -1:
#		X[conta,:] = Xbig[i,:]
#		conta = conta + 1

print('Blocked counter: ' + str(cm1) )
print('Y and X number of rows: ' + str(Y.shape[0]) + ', ' + str(X.shape[0])  )
print('First row of X: ' + str(X[0,:]))
print('Last row of X: ' + str(X[-1,:]))
#print(Y[0])
#print(Y[-1])
types = ['']

# if dataset == 1:
# 	Y = np.load('80_INITIAL_EPI_ENDO/Y_INITIAL_EPI_ENDO_998.npy')
# 	X = np.loadtxt('80_INITIAL_EPI_ENDO/LH_INITIAL_EPI_ENDO_998_10par.txt')
# 	xlabels = 'INITIAL_EPI_ENDO'
# 	# ranges = np.load('80_INITIAL_EPI_ENDO/ranges_INITIAL_EPI_ENDO.npy')
# 	labels = ['radius','depth','sigmaS','Int Bath','EHT thick','sigmaEHT','CP thick','sigmaCP','Contact Thick','Delta Thick']    # sono 10
# 	types = ['']
# elif dataset == 2:
# 	Y = np.load('81_TRANSMURAL/Y_TRANSMURAL_895.npy')
# 	X = np.loadtxt('81_TRANSMURAL/LH_TRANSMURAL_895_9par.txt')
# 	xlabels = 'TRANSMURAL'
# 	# ranges = np.load('81_TRANSMURAL/ranges_TRANSMURAL.npy')
# 	labels = ['radius','sigmaS','Int Bath','EHT thick','sigmaEHT','CP thick','sigmaCP','Contact Thick','Delta Thick']   # sono 9
# 	types = ['']
# elif dataset == 3:
# 	Y = np.load('82_TOTAL_BLOCK/Y_TOTAL_BLOCK_592.npy')
# 	X = np.loadtxt('82_TOTAL_BLOCK/LH_TOTAL_BLOCK_592_8par.txt')
# 	xlabels = 'TOTAL_BLOCK'
# 	# ranges = np.load('82_TOTAL_BLOCK/ranges_TOTAL_BLOCK.npy')
# 	labels = ['radius','Int Bath','EHT thick','sigmaEHT','CP thick','sigmaCP','Contact Thick','Delta Thick']   # sono 8
# 	types = ['']
# elif dataset == 4:
# 	Y = np.load('83_TOTAL_FIXED/Y_TOTAL_FIXED_587.npy')
# 	X = np.loadtxt('83_TOTAL_FIXED/LH_TOTAL_FIXED_587_7par.txt')
# 	xlabels = 'TOTAL_FIXED'
# 	labels = ['Int Bath','EHT thick','sigmaEHT','CP thick','sigmaCP','Contact Thick','Delta Thick']   # sono 7
# 	types = ['']
# elif dataset == 5:
# 	Y = np.load('84_INITIAL_ENDO_EPI/Y_INITIAL_ENDO_EPI_999.npy')
# 	X = np.loadtxt('84_INITIAL_ENDO_EPI/LH_INITIAL_ENDO_EPI_999_10par_explicit.txt')
# 	xlabels = 'INITIAL_ENDO_EPI'
# 	labels = ['radius','depth','sigmaS','Int Bath','EHT thick','sigmaEHT','CP thick','sigmaCP','Contact Thick','Delta Thick']    # sono 10
# 	types = ['']


#gpa.GridSearch(X,y)

for ww in range(len(types)):

	print('In corso iterazione ' + xlabels )
	saveFlag = 'y'
	gp, poly = gpa.Emulator(X,Y,saveFlag,xlabels,types[ww])

	# gpa.GridSearch(X,Y)

	# Load GP and Load Polynomial:
	#defaultName_Gps = 'GP_' + str(xlabels) + '_' + str(types[ww])
	#defaultName_poly = 'poly_' + str(xlabels) + '_' + str(types[ww])

	#NameGPtoLoad = defaultName_Gps 
	#with open(NameGPtoLoad + '.pickle', 'rb') as f:
	#	gp = pickle.load(f)

	#NamePoly = defaultName_poly 
	#with open(NamePoly + '.pickle', 'rb') as f:
	#	poly = pickle.load(f)

	# saveFlag = 'y'
	# gpa.GSA(gp,poly,ranges,labels,saveFlag,xlabels,types[ww])
