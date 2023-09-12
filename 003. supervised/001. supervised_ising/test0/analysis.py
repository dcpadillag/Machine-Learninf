from os import walk
import numpy as np
import matplotlib.pylab as plt

for (dirpath,dirnames,filenames) in walk('temp/.'):
    files=filenames
M=[]
X=[]
T=[]
files.sort()
for i in files:
    print(i)
    data=np.genfromtxt('temp/'+i)
    dataMag=data[:,1]
    dataX=data[:,2]
    T.append(float(i))
    Maverage=sum(dataMag[-200:])/200.
    Xaverage=sum(dataX[-200:])/200.
    M.append(Maverage)
    X.append(Xaverage)	


plt.plot(T,M,'.')
plt.show()

plt.plot(T,X,'.')
plt.show()
