# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:15:55 2021
Last updated on Mon Oct 16 2023
Refer to: http://gregorygundersen.com/blog/2019/08/13/bocd/
The main function have been modified.
Python version: 3.11.5
"""

import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.stats import norm
from   scipy.special import logsumexp
import pandas as pd


# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #    
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    T           = len(data)
    log_R       = -np.inf * np.ones((T+1, T+1))
    log_R[0, 0] = 0              # log 0 == 1
    pmean       = np.empty(T)    # Model's predictive mean.
    pvar        = np.empty(T)    # Model's predictive variance. 
    log_message = np.array([0])  # log 0 == 1
    log_H       = np.log(hazard)
    log_1mH     = np.log(1 - hazard)

    for t in range(1, T+1):
        # 2. Observe new datum.
        x = data[t-1]

        # Make model predictions.
        pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
        pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * model.var_params[:t])
        
        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        log_R[t, :t+1]  = new_log_joint
        log_R[t, :t+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R, pmean, pvar


# -----------------------------------------------------------------------------


class GaussianUnknownMean:
    
    def __init__(self, mean0, var0, varx):
        """Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx

# -----------------------------------------------------------------------------

def generate_data(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps


# -----------------------------------------------------------------------------

def plot_posterior(T, data, R):
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1,ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
    # Plot predictions.
    # ax1.plot(range(0, T), pmean, c='k')
    # _2std = 2 * np.sqrt(pvar)
    # ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    # ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=1e-7, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    hazard = 1/300  # the prior knowledge of CP frequency

    # where the prediction results(err or std) are saved
    data2 = pd.read_csv('/Users/houjiawen/Desktop/stderror.csv', header=None)  
    
    data2 = np.array(data2)[0,:]
    #data2=abs(data2)
    data2 = list(data2)
    maxlength = len(data2)
#    data2 = data2[:maxlength]
    leng = len(data2)
    meanx = np.mean(data2)
    varx = np.var(data2)
    var0 = varx
    mean0 = meanx
    curr_cp_num = 0

    t_ini = 0
    candidates = []
    max_cols = []
    model = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd(data2, model, hazard)

    while 1:
        last_tini = t_ini

        # for i in range(t_ini,leng-2):
        #      difR[i] = R[i,i-t_ini]-R[i+1,i+1-t_ini]
        # if t_ini>0:
        #     for j in range(0,t_ini-1):
        #         difR[j] = 0
        # max_difR = max(difR)
        # max_pos_row = np.argmax(difR)
        # R_max_row = R[max_pos_row+1,:max_pos_row - 15]
        # max_colR = max(R_max_row)
        # max_pos_col = np.argmax(R_max_row)
        # curr_cp = max_pos_row-max_pos_col
        for i in range(t_ini,leng-2):
            
            max_col = np.argmax(R[i,0:i-t_ini+1])
            if (max_col) != i-t_ini and R[i,max_col]>0.2:
                curr_cp = i-max_col
                max_cols = np.append(max_cols, max_col)
                break
            
        if (curr_cp - t_ini>=10):
            curr_cp_num += 1
            candidates = np.append(candidates, curr_cp)
            print('Number of change points: ',curr_cp_num)
        t_ini = i+1
        
        if t_ini-last_tini <= 0:
            break
        if maxlength-t_ini<=0.01*leng:
            break
        
    plot_posterior(leng, data2, R)
    if len(candidates)==0:
        print('No change points are detected !')
    else:
        print('Detected change points are at: ',candidates)
        
