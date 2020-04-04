import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

STDs = range(1, 4) # W
As = range(1, 4) # E
N = 1000
P0 = 0.8
P1 = 1 - P0

def findROC(C00, C01, C10, C11, beta, A, STD, Y, A_vec):
    eta = (C10-C00/C01-C11)*P0/P1*beta
    Y_dec= A/2 + STD**2*np.log(eta)/A
    Y1 = (Y > Y_dec)
    Y0 = (Y < Y_dec)
    # part 1 0.8 * 0.0296 (X > Y_dec) P(X > eta)
    # part 2 0.2 * 0.8123 (X < Y_dec) P(X+A < eta)
    
    # Two parts to the error: P(Deciding detected when not present) + P(Deciding not detected when present)
    theo_prob = (1-norm.cdf(Y_dec/STD))*P0 + (norm.cdf((Y_dec-A)/STD))*P1
    print("theoretical probability error: {}".format(theo_prob))
    print("measured probability error: {}".format(np.sum(np.logical_xor(A_vec, Y1))/N))
    
    # part b
    # TP FP
    TP = np.sum(A_vec*Y1)/N
    FP = np.sum(np.logical_not(A_vec)*Y1)/N
    TN = np.sum(np.logical_not(A_vec)*Y0)/N
    FN = np.sum(A_vec*Y0)/N
    
    return TP, FP
    
#for STD, A in zip(STDs, As):
for STD in STDs:
    for A in As:
        A_vec = np.random.uniform(size=1000) < P1
        X = np.random.normal(0, STD, N)
        Y = A*A_vec + X
        
        # MPE part a
        # C00 = 0 C11 = 0
        # C01 = 1 C10 = 1
        
        TPs = []
        FPs = []
        for beta in np.linspace(0, 2):
            TP, FP = findROC(C00=0,
                             C01=1.0,
                             C10=1.0,
                             C11=0,
                             beta=beta,
                             A=A,
                             STD=STD,
                             Y=Y,
                             A_vec=A_vec)
            TPs.append(TP)
            FPs.append(FP)
        plt.plot(FPs, TPs, 'o-')
plt.show()

# part c