# ********* Class of New Composite MMO Problems **************

# coded by  A. A.
# last update on 29-1-2024 by A. A.

import numpy as np
from ProblemMM import ProblemMM
pid=1; # problem ID
insNo=2; # problem instance No
dim=2; # dimensionality

problem=ProblemMM(pid,insNo,dim) #  creates the problem object

problem.form() #  calculates problem data

#  evalaute an arbitrary solution
X=np.ones(problem.dim)
print('X=',X)
f=problem.func_eval(X)
print('f(X)=',f)
    

    
                
                
                    





     
 
        

