# plots the landscape of the composite function
from ProblemMM import ProblemMM
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

pid=10 
pin=1 # problem instance No
dim=2 # dimensionality must be 2 for plotting

Ndiv=201 # mesh fineness for plotting
sc='log'  # scale of y axis: 'lin', 'log', 'or 'sqrt'
problem=ProblemMM(pid,pin,dim)
problem.form()
x1 = np.linspace(problem.lowBound,problem.upBound,Ndiv);
x2 = np.linspace(problem.lowBound,problem.upBound,Ndiv);

X1,X2=np.meshgrid(x1,x2)
z=np.zeros((Ndiv,Ndiv))
for k1 in np.arange(Ndiv):
    for k2 in np.arange(Ndiv):
        z[k1,k2]=problem.func_eval(np.array([x1[k1], x2[k2]]))
        
fig=plt.figure(1,figsize=(9,5))

if sc=='log':
    Z=np.log10(z-problem.globMinF);
    zlabel='$log_{10}(f-f_{min})$'
    
elif sc=='sqrt':
    Z=np.sqrt(z-problem.globMinF);
    zlabel='$\sqrt{f-f_{min}}$'

else:
    Z=z-problem.globMinF;
    zlabel='f-f_{min}'

ax=fig.add_subplot(1,2,1,projection="3d")
ax.plot_surface(X1, X2, Z.T, linewidth=0,  cmap=cm.viridis, antialiased=False)
ax.set_xlabel('x1');
ax.set_ylabel('x2');
ax.set_zlabel(zlabel);

ax=fig.add_subplot(1,2,2)
ax.contour(X1,X2,Z.T,30,cmap=cm.viridis)
ax.set_xlabel('x1');
ax.set_ylabel('x2');
ax.plot(problem.globMinX[:,0],problem.globMinX[:,1],'r*')

fig.tight_layout(pad=5)
