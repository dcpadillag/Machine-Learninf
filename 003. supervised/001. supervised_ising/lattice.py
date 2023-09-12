import numpy as np
import random
import sys

N=int(sys.argv[1])
lattice=np.empty((N,N))

for i in range(N):
	for j in range(N):
		lattice[i,j]=random.choice([-1,1])

np.save('lattice.npy',lattice)
