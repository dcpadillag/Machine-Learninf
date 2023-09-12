import numpy as np
import matplotlib.pyplot as plt
import sys

def sigmod(x):
	return 1/(1+np.exp(-x))

def dsigmod(x):
	return sigmod(x)*(1.-sigmod(x))

############MAIN##################
data=[[3., 1.5, 1],
      [2.,1.,0],
      [4.,1.5,1],
      [3.,1.,0],
      [3.5,.5,1],
      [2.,.5,0],
      [5.5,1.,1],
      [1.,1.,0]]

mistery=[float(sys.argv[1]),float(sys.argv[2])]

for i in range(len(data)):
	point=data[i]
	color='r'
	if point[2]==0:
		color='b'
	plt.scatter(point[0],point[1],c=color)

plt.scatter(mistery[0],mistery[1],c='yellow')
plt.title('before of learning')
plt.savefig('before.png')
plt.show()

##Neural Network

w1=np.random.rand()
w2=np.random.rand()
b=np.random.rand()

z=mistery[0]*w1+mistery[1]*w2+b
pred=sigmod(z)

'''
x=np.linspace(-5,5,100)
y=sigmod(x)
z=dsigmod(x)

plt.plot(x,y)
plt.plot(x,z)
plt.show()
'''

if pred>.5:
	mistery.append(1)
#	print 'The flower is red'
else:
	mistery.append(0)
#	print 'The flower is blue'

for i in range(len(data)):
	point=data[i]
	color='r'
	if point[2]==0:
		color='b'
	plt.scatter(point[0],point[1],c=color)

color='r'
if mistery[2]==0:
	color='b'
plt.scatter(mistery[0],mistery[1],c=color)
plt.title('first prediction')
plt.savefig('first.png')
plt.show()


########Training#############
#tstep=250000
tstep=int(sys.argv[3])*10000
learning_rate=0.2

costs=[]

for i in range(tstep):
	ri=np.random.randint(len(data))
	point=data[ri]

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


z=mistery[0]*w1+mistery[1]*w2+b
pred=sigmod(z)

if pred>.5:
	mistery[2]=1
#	print 'The flower is red'
else:
	mistery[2]=0
#	print 'The flower is blue'


for i in range(len(data)):
	point=data[i]

	z=point[0]*w1+point[1]*w2+b
	pred=sigmod(z)
	if pred>.5:
		point[2]=1
	else:
		point[2]=0
	color='r'
	if point[2]==0:
		color='b'
	plt.scatter(point[0],point[1],c=color)

color='r'
if mistery[2]==0:
	color='b'
plt.scatter(mistery[0],mistery[1],c=color)
plt.title('after of learning')
plt.savefig('after.png')
plt.show()

