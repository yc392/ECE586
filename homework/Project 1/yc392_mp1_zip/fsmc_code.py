import numpy as np
import numpy.matlib
import random
from scipy.linalg import null_space


# General function to expected hitting time for Exercise 2.1
def compute_Phi_ET(P, ns=100):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        ns {int} -- largest step to consider

    Returns:
        Phi_list {numpy.array} -- (ns + 1) x n x n, the Phi matrix for time 0, 1, ...,ns
        ET {numpy.array} -- n x n, expected hitting time approximated by ns steps ns
    '''
    a = np.shape(P)[0]
    b = np.shape(P)[1]
    Phi_list = np.zeros((ns+1,a,b))
    for m in range(0,ns+1):
        for i in range(0,a):
            for j in range(0,b):
                if(m==0):
                    if(i==j):
                        Phi_list[m][i][j] = 1
                else:
                    sum = 0
                    for k in range(0,a):
                        sum += P[i][k] * Phi_list[m-1][k][j]
                    Phi_list[m][i][j] = (i == j) + (1 - (i == j)) * sum
                #Phi_list[m][i][j] = ((np.mat(P) ** m).tolist())[i][j]
                
    ET = np.zeros((a,b))                              
    for i in range(0, a):
        for j in range(0,b):
            exp = 0
            for m in range(1,ns+1):
                exp += m * (Phi_list[m][i][j]-Phi_list[m-1][i][j])
            ET[i][j] = exp
    # Add code here to compute following quantities:
    # Phi_list[m, i, j] = phi_{i,j}^{(m)} = Pr( T_{i, j} <= m )
    # ET[i, j] = E[ T_{i, j} ] ~ \sum_{m=1}^ns m Pr( T_{i, j} = m )
    # Notice in python the index starts from 0

    return Phi_list, ET
                                     
# General function to simulate hitting time for Exercise 2.1
def simulate_hitting_time(P, states, nr):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        states {list[int]} -- the list [start state, end state], index starts from 0
        nr {int} -- largest step to consider

    Returns:
        T {list[int]} -- a size nr list contains the hitting time of all realizations
    '''
    i = states[0]                                 
    j = states[1]
    phi_lst = compute_Phi_ET(P)[0]                                 
    prob_lst = []                                 
    for m in range(1,101):                                 
        prob_lst.append(phi_lst[m][i][j] - phi_lst[m-1][i][j])                             
    prob_lst.append(1 - sum(prob_lst))                                                                      
    T = np.random.choice(range(1,102), size = nr, p = prob_lst).tolist()
    # Add code here to simulate following quantities:
    # T[i] = hitting time of the i-th run (i.e., realization) of process
    # Notice in python the index starts from 0
                                                              
    return T



# General function to approximate the stationary distribution of a Markov chain for Exercise 2.4
def stationary_distribution(P):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain

    Returns:
        pi {numpy.array} -- length n, stationary distribution of the Markov chain
    '''

    # Add code here: Think of pi as column vector, solve linear equations:
    #     P^T pi = pi
    #     sum(pi) = 1
    shape = np.shape(P)[0]
    I = np.identity(shape)
    T = np.transpose(I-P)
    ns = null_space(T)
    pi = ns / sum(ns)[0]
                                     
    return pi