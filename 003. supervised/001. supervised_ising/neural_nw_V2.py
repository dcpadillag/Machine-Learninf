from os import walk
import numpy as np
import matplotlib.pylab as plt
import sys

def sigmod(x):
	return 1/(1+np.exp(-x))

def softmax(a):
	return np.array([(np.exp(a[0]))/(np.exp(a[0])+np.exp(a[1])),(np.exp(a[1]))/(np.exp(a[0])+np.exp(a[1]))])

def dsigmod(x):
	return sigmod(x)*(1.-sigmod(x))

#################LOAD_INPUT###############
file_name=sys.argv[1]
config=(np.load('data/'+file_name+'.npy'))

data=np.genfromtxt('temp/'+file_name)
dataMag=data[:,1]
M=sum(dataMag[-200:])/200.
##########################################
#############NETWORK_INI########################
if int(sys.argv[3])==0:
	N=np.size(config)
	Nf=int(sys.argv[2])  #Number of neurons in the hidden layer

	w_hidden=np.random.rand(Nf,N)
	b_hidden=np.random.rand(Nf,1)

	w_out=np.random.rand(2,Nf)
	b_out=np.random.rand(2,1)
else:
	N=np.size(config)
	Nf=int(sys.argv[2])  #Number of neurons in the hidden layer

	w_hidden=np.load('parameters/w_hidden.npy')
	b_hidden=np.load('parameters/b_hidden.npy')

	w_out=np.load('parameters/w_out.npy')
	b_out=np.load('parameters/b_out.npy')
#############################################
####################TRAINING#################
#tstep=250000
#tstep=int(sys.argv[2])*10
learning_rate=0.2

costs=[]

x_in=np.reshape(config,[N,1])

a_hidden=np.matmul(w_hidden,x_in)+b_hidden
z_hidden=sigmod(a_hidden)

a_out=np.matmul(w_out,z_hidden)+b_out
y_out=softmax(a_out)

cost=np.log(y_out[0])+np.log(1.-y_out[1])
costs.append(cost)

######dcost/dw_hidden#################################################
dzhidden_dwhidden=np.empty([Nf,N])

dzhidden_dahidden=dsigmod(a_hidden)

for i in range(Nf):
#	dahidden_dwhidden[i,:]=b_hidden[i]+x_in.transpose()
	dzhidden_dwhidden[i,:]=dzhidden_dahidden[i]*(b_hidden[i]+x_in.transpose())


daout_dzhidden=w_out

daout_dzhidden[0,:]=daout_dzhidden[0,:]+b_out[0]
daout_dzhidden[1,:]=daout_dzhidden[1,:]+b_out[1]

dyout_daout=np.matrix([[y_out[0][0]*(1.-y_out[0][0]),-y_out[0][0]*y_out[1][0]],[-y_out[0][0]*y_out[1][0],y_out[1][0]*(1.-y_out[1][0])]])

dcost_dyout=np.array([1./y_out[0][0],1./(y_out[1][0]-1.)])


dcost_dwhidden=np.empty([Nf,N])

for i in range(Nf):
	for j in range(2):
		for k in range(2):
			dcost_dwhidden[i,:]=dcost_dwhidden[i,:]+dcost_dyout[j]*dyout_daout[j,k]*daout_dzhidden[k,i]*dzhidden_dwhidden[i,:]

w_hidden=w_hidden-learning_rate*dcost_dwhidden 
######################################################################
######dcost/db_hidden#################################################

######################################################################

np.save('parameters/w_hidden.npy',w_hidden)
np.save('parameters/b_hidden.npy',b_hidden)
np.save('parameters/w_out.npy',w_out)
np.save('parameters/b_out.npy',b_out)


'''

dcost_dyout=np.array([1./y_out[0],1./(y_out[1]-1.)])

dyout0_daout0=y_out[0]*(1.0-y_out[0])
dyout0_daout1=-y_out[0]*y_out[1]

dyout1_daout0=-y_out[1]*y_out[0]
dyout1_daout1=y_out[1]*(1.0-y_out[1])

daout0_dwout=???
daout0_dbout=np.matmul(w_out,z_hidden.transpose()).transpose()

daout0_dzhidden=w_out

dzhidden_dahidden=dsigmod(a_hidden)

dahidden_dwhidden=??
dahidden_dbhidden=np.matmul(w_hidden,x_in)

dcost_dwhidden=(dcost_dyout0*dyout0_) + ()
'''

'''
z=point[0]*w1+point[1]*w2+b
pred=sigmod(z)
target=point[2]
cost=(pred-target)**2

costs.append(cost)
dcost_dpred=2*(pred-target)
dpred_dz=dsigmod(z)
dcost_dz=dcost_dpred*dpred_dz

dz_dw1=point[0]
dz_dw2=point[1]
dz_db=1.

dcost_dw1=dcost_dz*dz_dw1
dcost_dw2=dcost_dz*dz_dw2
dcost_db=dcost_dz*dz_db

w1=w1-learning_rate*dcost_dw1
w2=w2-learning_rate*dcost_dw2
b=b-learning_rate*dcost_db
	
if i%1000==0:
	cost_sum=0
	for j in range(len(data)):
		point=data[ri]

		z=point[0]*w1+point[1]*w2+b
		pred=sigmod(z)
	
		target=point[2]
		cost_sum=cost_sum+(pred-target)**2
	costs.append(cost_sum/len(data))

plt.plot(costs)
plt.savefig('cost.png')
plt.show()
'''





