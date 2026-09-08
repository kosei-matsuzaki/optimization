# ********* Class of New Composite MMO Problems **************

# coded by  A. A.
# last update on 29-1-2024 by A. A.

import numpy as np
from BasicFun import *
from UtilityMethod import *
from scipy.spatial.distance import cdist
class ProblemMM:
    """The main class. It creates an object which determine the problem"""
    __slots__ = [
        'pid','funID','insNo','dim','maxEval','lowBound','upBound','hardGO','hardNU',
        'rotAngleLocalCoef', 'rotAngleGlobalCoef','usedEval','lambda0','sigmaW',
        'dMin','maxEvalCoef','numGlobMin','globMinX','globMinF','globMinHard',
        'globMinRangeCoef','rotMatGlobal','rotMat','nichRad','crowdBasinInd','specialNo'
        ] 
    def __init__(self,pid,insNo,dim): # requires functions ID and problem dimensionality    
        data=np.loadtxt(r'data/pidData.csv',delimiter=',');
        self.pid=pid; #problem ID
        self.funID=int(data[pid-1,0]); # function ID (different from problem ID)        
        self.insNo=insNo; # instance No
        self.dim=dim; # dimensionality
        self.lowBound=-5; #the lower bound of the search space
        self.upBound=5; # the upper bound of the search space
        self.numGlobMin=int(data[pid-1,1]); # (scalar) number of global minima
        self.hardGO=data[pid-1,2:4]; # a number in [0,1] that specifies the hardness from global optimization perspective
        self.hardNU=data[pid-1,4]; # a non-negative Real number specifying the non-uniformity in the distribution of global minima 
        self.rotAngleLocalCoef=np.pi/2 # standard deviation for the angle of rotation of basic functions
        self.rotAngleGlobalCoef=np.pi  # standard deviation for the global rotation matrix angle
        self.lambda0=data[pid-1,5]; # for scaling the search range
        self.dMin=0.3*self.dim**.5; # distance threshold between global minima
        self.usedEval=0 # used evaluation so far   
        self.sigmaW=.5  # controls the impact extent of a basic function  
        self.maxEvalCoef=50000; # the coefficient for the evaluation budget
        self.globMinX=None # (matrix) global minima of the static problem (or at time step #0 if the problem is dynamic)
        self.globMinF=None # the global minimum value (scalar) 
        self.globMinHard=None # hardness of finding each global minimum from GO perspective        
        self.globMinRangeCoef=0.9; # global minima are inside this fraction of each dimensionality, excluding close to bounds regions        
        self.rotMatGlobal=None; # (matrix) rotation matrix 
        self.rotMat=None; # Rotation matrix for each basic function        
        self.nichRad=None; # Niching radius
        self.crowdBasinInd=None; # index of the global minimum that other solutions are redistributed wrt
        self.specialNo=None; # problem special number used for reading random numbers from CSV files
    def __str__(self): # display the problem
         output=('\npid:'+'\n    pid = ' + str(self.pid) +
                 '\n\tpin = ' + str(self.insNo) +
                 '\n\tdim = ' + str(self.dim) +                 
                 '\n\tlowBound = ' + str(self.lowBound) + 
                 '\n\tupBound = ' + str(self.upBound) + 
                 '\n\tmaxEval = ' + str(self.maxEval) +
                 '\n\tusedEval = ' + str(self.usedEval)               
                 )
         return output

    def form(self): # create problem data
        numUniform=np.loadtxt(r'data/num-uniform.csv',delimiter=',',dtype=float); # array of uniformly distributed random numbers in (0,1)
        numNormal=np.loadtxt(r'data/num-normal.csv',delimiter=',',dtype=float); # array of standard normal numbers  
        sequences=np.loadtxt(r'data/sequences.csv',delimiter=',',dtype=int); # matrix, 240 x 5101
        fstarData=np.loadtxt(r'data/fstarData.csv',delimiter=',',dtype=float); # global minimum values - 1-D array

        # determine the sequence 
        indUni=1;indNorm=1; # read next random numbers at this index
        self.specialNo=15*(self.pid-1)+self.insNo; # index number for this pid and insNo
        useSeq=sequences[self.specialNo-1,:]-1; # use this sequence of random numbers

        # choose random numbers to create locations of global minima
        temp=numUniform[useSeq[indUni-1:indUni-1+5*self.dim*self.numGlobMin]]; # 5 times random numbers in (0,1)
        indUni=indUni+5*self.dim*self.numGlobMin; # for reading subsequent numbers 
        randX=temp.reshape(self.numGlobMin*5,self.dim); # solutions from which global minima are selected
        
        # set Xref for redistribution 
        self.crowdBasinInd=int(np.ceil(numUniform[useSeq[indUni-1]]*self.numGlobMin)); # index of Xref is selected randomly
        indUni=indUni+1; # index of used random number with uniform distribution

        # set global minima
        uniformX,tmp=UtilityMethod.keep_farthest(randX,self.numGlobMin); # select farthest ones
        uniformX=uniformX[0:self.numGlobMin,:]*self.globMinRangeCoef+(1-self.globMinRangeCoef)/2; # uniform distribution in reduced range [0,1]
        uniformX = uniformX*(self.upBound-self.lowBound)+self.lowBound; # uniform distribution in search space
        self.globMinX,tmp=UtilityMethod.redist_glob_min(uniformX,uniformX[self.crowdBasinInd-1,:],self.hardNU,self.dMin); # non-uniform distribution
       
        # set each global minimum hardness:             
        ind0=np.argsort(numUniform[useSeq[indUni-1:indUni-1+self.numGlobMin]]); # use random numbers to sort out the hardness
        indUni=indUni+self.numGlobMin;
        coef=(ind0)/(self.numGlobMin-1); # This is numGlobMin uniformly distributed numbers in [0,1] with random order
        self.globMinHard= coef* (self.hardGO[1]-self.hardGO[0])+self.hardGO[0];

        # set nichRad
        if self.numGlobMin==1:
            self.nichRad=5*np.sqrt(self.dim);
        else:
            tmp=cdist(np.atleast_2d(self.globMinX),np.atleast_2d(self.globMinX));
            tmp=tmp+np.max(tmp)*np.eye(self.numGlobMin);
            self.nichRad=np.min(tmp,axis=0)/2.0; 
        
        # set other attributes
        self.maxEval=int(np.round(self.maxEvalCoef*self.dim));
        self.globMinF=fstarData[self.funID-1];

        # get random Normal numbers for creating Rotation matrices
        temp0=2*self.dim*(self.numGlobMin+1); # number of random Normal number 
        tempUV=numNormal[useSeq[indNorm-1:indNorm-1+temp0]]; # get first temp0 numbers of the sequence of Normal numbers
        indNorm=indNorm+temp0;
        allUV=tempUV.reshape(2*(self.numGlobMin+1),self.dim) # each row is a vector of size D
        allAngleData=numNormal[useSeq[indNorm-1:indNorm-1+self.numGlobMin+1]]; # 1-D array of size numGlobMin+1
        indNorm=indNorm+self.numGlobMin+1;
        # create the global rotation matrix
        self.rotMat=[' ']*self.numGlobMin;
        if not (self.rotAngleGlobalCoef==0) and self.dim>1: # calculate global rotation matrices
            rotAngleGlobal=self.rotAngleGlobalCoef*allAngleData[0];
            u0=allUV[0,:];
            v0=allUV[1,:];
            self.rotMatGlobal=UtilityMethod.gen_rot_mat_pseudo(u0,v0,rotAngleGlobal);
        else:
            self.rotMatGlobal=np.eye(self.dim);

        # create individual rotation matrix
        if self.rotAngleLocalCoef>0 and self.dim>1:                
            for k in np.arange(1,self.numGlobMin+1):
                rotAngleLocal=self.rotAngleLocalCoef*allAngleData[k]; # start from the second element
                u0=allUV[2*k,:]; # start from the third element
                v0=allUV[2*k+1,:]; # start from the fourth element
                R0=UtilityMethod.gen_rot_mat_pseudo(u0,v0,rotAngleLocal);
                self.rotMat[k-1]=self.rotMatGlobal @ R0;
        else:
            for k in np.arange(1,self.numGlobMin+1):
                self.rotMat[k-1]=self.rotMatGlobal;
        pass
        
    # objective function accepts a matrix where each row is a solution
    def func_eval(self,x0) :
        x=np.atleast_2d(x0)
        N=x.shape[0];
        f=np.zeros(N);
        for k in np.arange(1,N+1):
            f[k-1]=self.func_eval_single(x[k-1,:])+self.globMinF;
        return f

    def func_eval_single(self,x):
        F=np.zeros(self.numGlobMin); # fitness values
        for k in np.arange(1,self.numGlobMin+1):
            shift=self.globMinX[k-1,:];
            xRot=(x-shift) @ self.rotMat[k-1]
            F[k-1]=BasicFun.evaluate(xRot/self.lambda0,self.globMinHard[k-1],self.funID);
            pass
        # Calculate weights
        dis=cdist(np.atleast_2d(x),np.atleast_2d(self.globMinX)).ravel();
        normDis2=(dis/(self.sigmaW*self.nichRad))**2;
        normDis2min=np.min(normDis2);
        if normDis2min<=1:
            C0=0;
        else:
            C0=1-normDis2min;
        W=np.exp(-normDis2-C0);
        maxW=np.max(W);
        term= np.abs(W-maxW)<1e-14;
        W=W*(1-maxW**10) * (1-term) + W * term;
        W=W/np.sum(W);
        f=np.sum(F*W);            
        self.usedEval=self.usedEval+1;
        return f
        
if __name__=='__main__': # a simple test of this class
    pid=1;insNo=2;dim=2
    problem=ProblemMM(pid,insNo,dim)
    problem.form()
    X=np.ones(problem.dim)
    print('X=',X)
    f=problem.func_eval(X)
    print('f(X)=',f)
    

    
                
                
                    





     
 
        

