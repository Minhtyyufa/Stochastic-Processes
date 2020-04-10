import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

STDs = range(1, 3) # W
As = range(1, 3) # E
zs = [2, 10]
N = 1000
P0 = 0.8
P1 = 1 - P0


def findROC(beta, A, STD, Y, A_vec):
    Y_dec = beta
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
        C00 = 0 
        C01 = 1
        C10 = 1
        C11 = 0
        
        TPs = []
        FPs = []
        for beta in np.linspace(-8, 8):
            TP, FP = findROC(beta=beta,
                             A=A,
                             STD=STD,
                             Y=Y,
                             A_vec=A_vec)

            TPs.append(TP)
            FPs.append(FP)
        
        # MLE Y_dec
        eta = ((C10-C00)*P0)/((C01-C11)*P1)
        Y_dec= A/2 + STD**2*np.log(eta)/A
        TP, FP = findROC(beta=Y_dec,
                         A=A,
                         STD=STD,
                         Y=Y,
                         A_vec=A_vec)


        # part c
        C10 = 1
        C01 = 10
        eta = ((C10-C00)*P0)/((C01-C11)*P1)
        Y_dec= A/2 + STD**2*np.log(eta)/A
        TP_changed_cost, FP_changed_cost = findROC(beta=Y_dec,
                         A=A,
                         STD=STD,
                         Y=Y,
                         A_vec=A_vec)

        plt.plot(FPs, TPs, 'o-')
        # MLE
        plt.plot(FP, TP, '*', markersize=16)
        # Part C adjusted cost MLE
        plt.plot(FP_changed_cost, TP_changed_cost, '^', markersize=16)
plt.show()


# Part D
P1s = np.linspace(0.01, 0.99, 100)
expecteds = []
A = 1
STD = 1
C10 = 1
C01 = 10
C00 = 0
C11 = 0
for P1 in P1s:
    A_vec = np.random.uniform(size=1000) < P1
    X = np.random.normal(0, STD, N)
    Y = A*A_vec + X
    
    eta = ((C10-C00)*(1-P1))/((C01-C11)*P1)
    
    Y_dec= A/2 + STD**2*np.log(eta)/A
    Y1 = (Y > Y_dec) 
    Y0 = (Y < Y_dec)
    TP = np.sum(A_vec*Y1)/N
    FP = np.sum(np.logical_not(A_vec)*Y1)/N
    TN = np.sum(np.logical_not(A_vec)*Y0)/N
    FN = np.sum(A_vec*Y0)/N

    # C00 = 0, C11 = 0
    expecteds.append(FP*C01 + FN*C10)
plt.plot(P1s, expecteds, '-o')
plt.show()

# E

# beta = +/- sqrt(gamma)
def findROC2(beta, A, STD, STDz, Y, A_vec):
    Y_dec = beta+A
    Y1 = (np.abs(Y-A) < Y_dec-A) 
    Y0 = (np.abs(Y-A) > Y_dec-A)
    # part 1 0.8 * 0.0296 (X > Y_dec) P(X > eta)
    # part 2 0.2 * 0.8123 (X < Y_dec) P(X+A < eta) 
    #Gamma_prime=(2*(STD**2*STDz**2)/(STD**2-STDz**2))*np.log(STD*eta/STDz)
    # Two parts to the error: P(Deciding detected when not present) + P(Deciding not detected when present)

    theo_prob = 2*(1-norm.cdf(Y_dec, scale=STDz, loc=A))*P0 + (norm.cdf(Y_dec, scale=STD, loc=A)-(1-norm.cdf(Y_dec, loc=A, scale=STD)))*P1

    print("theoretical probability error: {}".format(theo_prob))
    print("measured probability error: {}".format(np.sum(np.logical_xor(A_vec, Y1))/N))
    
    # part b
    # TP FP
    TP = np.sum(A_vec*Y1)/N
    FP = np.sum(np.logical_not(A_vec)*Y1)/N
    TN = np.sum(np.logical_not(A_vec)*Y0)/N
    FN = np.sum(A_vec*Y0)/N
    
    return TP, FP
    
P1 = .8
P0 = .2
    
for z in zs:
    for STD in STDs:
        for A in As:
            present = np.random.uniform(size=1000) < P1
            X = np.random.normal(0, STD, N)
            Z = np.random.normal(0, STD*z, N)
            Y = A + X * present + Z * np.logical_not(present)
            
            # MPE part a
            C00 = 0 
            C01 = 1
            C10 = 1
            C11 = 0
            
            TPs = []
            FPs = []
            TPzs = []
            FPzs = []
            for beta in np.linspace(0, 8):
                TP, FP = findROC2(beta=beta,
                                 A=A,
                                 STD=STD,
                                 STDz=STD*z,
                                 Y=Y,
                                 A_vec=present)
    
                TPs.append(TP)
                FPs.append(FP)

            plt.plot(FPs, TPs, 'o-')
            plt.plot(FPzs, TPzs, '--')
plt.show()