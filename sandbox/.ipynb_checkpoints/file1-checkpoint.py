import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''MAKE PRIOR'''
dist_prior = np.random.poisson(60,1000)
dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
dist_prior_event_probs = dist_prior_event_probs.sort_index()
plt.hist(dist_prior)
plt.hist(dist_prior_event_probs)

def prob_t_tot(x,dist):
    '''
    x = t_total
    dist is the right distribution and should be a pd.Series
    returns the event probability of t_total as a discrete event, not integrated
    NOTE, the ifelse control should really interpolate but for now it doesnt.
    '''
    if x in dist.index:
        a = dist[x]
    else:
        a = np.float64(0)
    #NOTE, THISFU  
    return a
#EG
prob_t_tot(34,dist_prior_event_probs)

def median_of_dist(dist):
    '''
    dist is a pd.Series
    returns the median index value which should be
    the value of the poisson
    '''
    return dist.loc[dist.cumsum()>0.5].index[0]
#EG
median_of_dist(dist_prior_event_probs)


def compute_p_t(t,t_total_max,dist):
    '''
    This integrates from t to t_total_max to return p(t),
    over this range, we integrate the following sum
    dist is the prior
    '''
    catch = np.array([])
    for i in range(t,t_total_max):
        a = 1/i
        b = prob_t_tot(i,dist)
        catch = np.append(catch,a*b)

    return catch.sum()

def compute_p_t_2(t,dist):
    '''
    This integrates from t to t_total_max to return p(t),
    over this range, we integrate the following sum
    dist is the prior
    '''
    vect_t_total = dist.index
    catch = np.array([])
    for i in vect_t_total:
        a = 1/i
        b = prob_t_tot(i,dist)
        catch = np.append(catch,a*b)

    return catch.sum()
'''EXAMPLE
provide context, 'if 10 days into an epidemic wave (t=10), how many days total (t_total) will be until the peak(or end)'
want to generate the distribution of t_total|t over range of t_total;
t_total drives all the variables, so this function will take in an vector t_total and compute and collect once per
index of t_total
'''


def compute_posterior(t,dist):
    '''
    dist is the ordered probablility lookup table fro each event t_total
    t_total is what we are computing the distribution over
    it is a np.array, ordered
    
    returns np.array, ordered as the 
    NOTE: using index of prior doesn't cover, necessarily, all
    possible values of t_total. Will fix later.
    '''
    print('t is: ',t)
    vect_t_total = dist.index
    print('t_total_range is: ',vect_t_total)
    
    catch = np.array([])
    
    for i in vect_t_total:
        print('t_total is: ',i)
        #prior
        prior = prob_t_tot(i,dist)
        likelihood = 1/i 
        p_t = compute_p_t(t,vect_t_total[-1],dist)
        
        catch = np.append(catch,prior*p_t*likelihood)
        
    return catch


x = compute_posterior(70,dist_prior_event_probs)
x = pd.Series(x,index=dist_prior_event_probs.index)
x.sum()
x_adj = x*4000
x_adj.sum()
median_of_dist(dist_prior_event_probs)
median_of_dist(x_adj)

def compute_decision():