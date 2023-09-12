from os import walk
import numpy as np
import matplotlib.pylab as plt
import sys

def sigmod(x):
	return 1/(1+np.exp(-x))

def softmax(a):
	return np.array([(np.exp(a[0,0]))/(np.exp(a[0,0])+np.exp(a[0,1])),(np.exp(a[0,1]))/(np.exp(a[0,0])+np.exp(a[0,1]))])

def dsigmod(x):
	return sigmod(x)*(1.-sigmod(x))

############LOAD_INPUT######################
for (dirpath,dirnames,filenames) in walk('data/.'):
    files=filenames

config=[]

for i in files:
    config.append(np.load('data/'+i))

for (dirpath,dirnames,filenames) in walk('temp/.'):
    files=filenames

M=[]

for i in files:
    print(i)
    data=np.genfromtxt('temp/'+i)
    dataMag=data[:,1]
    Maverage=sum(dataMag[-200:])/200.
    M.append(Maverage)

############################################
#############NETWORK_INI########################
N=np.size(config[0])
Nf=int(sys.argv[1])  #Number of neurons in the hidden layer

w_hidden=np.random.rand(Nf,N)
b_hidden=np.random.rand(1,Nf)

w_out=np.random.rand(2,Nf)
b_out=np.random.rand(2,1)
#############################################
####################TRAINING#################
#tstep=250000
tstep=int(sys.argv[2])*10
learning_rate=0.2

costs=[]

for i in range(tstep):
	index=np.random.randint(len(config))
	x_in=np.reshape(config[index],N)
	m=M[index]
	a_hidden=np.matmul(w_hidden,x_in)+b_hidden
	
	z_hidden=sigmod(a_hidden)
	print z_hidden

	a_out=np.matmul(w_out,z_hidden.transpose())+b_out
	a_out=a_out.transpose()

	y_out=softmax(a_out)
#	R=y_out[0]/y_out[1]
	cost=np.log(y_out[0])+np.log(1.-y_out[1])
	costs.append(cost)

	dcost_dyout0=1./y_out[0]
	dcost_dyout0=1./(y_out[1]-1.)

	dyout0_daout0=
	dyout0_daout1=

	dyout1_daout0=
	dyout1_daout1=

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
