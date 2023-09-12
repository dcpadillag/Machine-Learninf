import numpy as np
import matplotlib.pyplot as plt
import sys, random

def H(lattice0,i1,j1):
	i_r=i1+1
	i_l=i1-1
	j_u=j1+1
	j_d=j1-1

########PBC################
	if j_d<0: j_d=N-1
	if j_u==N: j_u=0
	if i_l<0: i_l=N-1
	if i_r==N: i_r=0
###########################
    
	return -J*lattice0[i1,j1]*(lattice0[i1,j_u]+lattice0[i1,j_d]+lattice0[i_r,j1]+lattice0[i_l,j1])

def metropolis(lattice0):
	i0=np.random.random_integers(0,N-1)
	j0=np.random.random_integers(0,N-1)
	Eold=H(lattice0,i0,j0)
	Enew=-Eold
	dE=(Enew-Eold)	
	if dE<0.:
		lattice0[i0,j0]=-lattice0[i0,j0]
	else:
		r=np.random.rand(1)[0]
		if r<np.exp(-dE/kbT):
			lattice0[i0,j0]=-lattice0[i0,j0]

def neighbors(j0):
	i1, j1=j0[0], j0[1]
	i_r=i1+1
	i_l=i1-1
	j_u=j1+1
	j_d=j1-1
########PBC################
	if j_d<0: j_d=N-1
	if j_u==N: j_u=0
	if i_l<0: i_l=N-1
	if i_r==N: i_r=0
###########################

	return [(i_r,j1),(i_l,j1),(i1,j_u),(i1,j_d)]
	
def wolff(lattice0):
	p=1.-np.exp(-2.*J/kbT)

	i0=np.random.random_integers(0,N-1)
	j0=np.random.random_integers(0,N-1)
	stack, cluster= [(i0,j0)], [(i0,j0)]
	while stack!=[]:
		j=random.choice(stack)
		nn=neighbors(j)
		for l in nn:
			r=np.random.rand(1)[0]
			if ((lattice[j[0],j[1]]==lattice[l[0],l[1]]) and (l not in cluster) and (r<p)):
				stack.append(l)
				cluster.append(l)
		stack.remove(j)

	Eold=H(lattice0,i0,j0)

	copy=lattice0
	for j in cluster:
		copy[j[0],j[1]]=-1*copy[j[0],j[1]]
	Enew=H(copy,i0,j0)

	dE=(Enew-Eold)
	if dE<0.:
		lattice0=copy
	else:
		r=np.random.rand(1)[0]
		if r<np.exp(-dE/kbT):
			lattice0=copy

####################MAIN######################33
kbT=float(sys.argv[1])/10
J=1.0
N=int(sys.argv[2])
lattice=np.load('lattice.npy')

#pl=plt.matshow(lattice,vmin=-1,vmax=1)
#plt.colorbar(pl)
#plt.savefig('1.png')
#plt.show()

NMS=N*N
magnetization=open("temp/"+str(kbT),"w")
for i in range(NMS):
	wolff(lattice)
#	metropolis(lattice)
	m=abs(np.sum(lattice)/NMS)
	lattice_sq=lattice**2
	m2=np.sum(lattice_sq)/NMS
	X=(m2-m**2)/kbT
	magnetization.write(str(i)+"\t"+str(m)+"\t"+str(X)+"\n")

#pl=plt.matshow(lattice,vmin=-1,vmax=1)
#plt.colorbar(pl)
#plt.savefig('gif/'+str(kbT)+'.png')

np.save('data/'+str(kbT),lattice)



