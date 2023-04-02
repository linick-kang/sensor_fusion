import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.animation
import math


pi = math.pi
time = np.arange(1,10,0.1)
s = len(time) # steps
gt = np.zeros((s,2)) #ground truth, 2 line for x,y coords

gt[:,0] = np.sin(time*2*pi/10)*10
gt[:,1] = 10*np.exp(time*-1/10)

det = np.zeros((s,2)) #detected coords
det[:,0] = gt[:,0] + (np.random.rand(s) + np.array([-0.5 for _ in range(s)])) * 0.2
det[:,1] = gt[:,1] + (np.random.rand(s) + np.array([-0.5 for _ in range(s)])) * 0.3

plt.plot(gt[:,0],gt[:,1], color='black',  linestyle='solid')
plt.scatter(det[:,0],det[:,1],c = 'blue')

"EKF part"
#Predict
A = np.matrix([[1,0],[0,1.2]])
xhat_init = np.matrix(det[0]).T
R = np.matrix([[0.04,0],[0,0.09]])
C = np.matrix([[1,0],[0,1]])
xhat = [xhat_init]
pk = [np.matrix([[1,0],[0,1]])]

count = 0
res = np.zeros((s,2))
while count < s:
    i = len(xhat) - 1
    xhat.append(np.dot(A,xhat[i]))
    pk.append(A*pk[i]*A.T)
    
    #Updata
    gk = pk[i] * C.T * np.linalg.inv((C * pk[i] * C.T + R))
    zk = np.matrix(det[i]).T
    xhat[i] = xhat[i] + gk * (zk - C * xhat[i])
    pk[i] = (np.matrix(np.identity(2)) - gk * C) * pk[i]
    res[count,0] = float(xhat[i][0])
    res[count,1] = float(xhat[i][1])
    count += 1
    
plt.plot(res[:,0],res[:,1], color='red',  linestyle='dashed')
    






