import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''MAKE PRIOR'''
#GENERATOR
N = 1000
dist_prior = np.random.poisson(75,N)
#dist_prior = np.random.exponential(15,N)
#dist_prior = np.round(dist_prior).copy()
print(dist_prior)
plt.hist(dist_prior,bins=30)

#COMPUTE PROBS FOR EACH EVENT
dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
dist_prior_event_probs = dist_prior_event_probs.sort_index()
plt.subplot(2,3,1)
plt.hist(dist_prior_event_probs,bins=30)
plt.subplot(2,3,2)
plt.plot(dist_prior_event_probs)
plt.subplot(2,3,3)
plt.hist(dist_prior,bins=30)
#SUMS TO ONE
dist_prior_event_probs.sum()

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

def n_t_tot(x,dist,n):
    '''
    x=t_total
    dist same as prob_t_tot
    n is N from generating dist.
    returns the N of t_total from the generating distribution
    '''
    if x in dist.index:
        a = dist[x]*n
    else:
        a = np.float64(0)
    #NOTE, THISFU  
    return a
#EG
n_t_tot(70,dist_prior_event_probs,N)

def median_of_dist(dist):
    '''
    dist is a pd.Series
    returns the median index value which should be
    the value of the poisson
    '''
    return dist.loc[dist.cumsum()>0.5].index[0]
#EG
median_of_dist(dist_prior_event_probs)

def compute_p_t(t,dist):
    '''
    t is for current problem
    dist is dist_prior_event_probs, will be pd series.
    returns a scalar numpy.float64
    '''
    subset = dist_prior_event_probs.iloc[dist_prior_event_probs.index>=t]
    #print('subset',subset)
    one_over_t_total = pd.Series(1/subset.index,index=subset.index) 
    #print('one over t tot',one_over_t_total)
    
    return (subset*one_over_t_total).sum()
#EG
compute_p_t(5,dist_prior_event_probs)
compute_p_t(75,dist_prior_event_probs)
'''compute_p_t LOGIpC'''
dist_prior_event_probs.index
one_over_ttot = pd.Series(1/dist_prior_event_probs.index,index=dist_prior_event_probs.index)





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
        likelihood = (1/i)*(n_t_tot(i,dist_prior_event_probs,N)) 
        p_t = compute_p_t(t,dist)
        likelihood_ratio = (1/i)/p_t
        print('likelihood, p_t, likelihood_ratio, prior, post: ',likelihood, p_t, likelihood_ratio, prior, likelihood_ratio*prior)
        catch = np.append(catch,likelihood_ratio*prior)
        
    return catch

def compute_posterior_exp(t,dist):
    '''
    dist is the ordered probablility lookup table fro each event t_total
    t_total is what we are computing the distribution over
    it is a np.array, ordered
    
    returns np.array, ordered as the 
    NOTE: using index of prior doesn't cover, necessarily, all
    possible values of t_total. Will fix later.
    '''
    print('t is: ',t)
    vect_t_total = dist.index+1
    print('t_total_range is: ',vect_t_total)
    
    catch = np.array([])
    
    for i in vect_t_total:
        print('t_total is: ',i)
        #prior
        prior = prob_t_tot(i,dist)
        likelihood = 1/i 
        p_t = compute_p_t(t,dist)
        likelihood_ratio = (1/i)/p_t
        print('likelihood, p_t, likelihood_ratio, prior, post: ',likelihood, p_t, likelihood_ratio,prior,likelihood_ratio*prior)
        catch = np.append(catch,likelihood_ratio*prior)
        
    return catch


#RUN ABOVE
x = compute_posterior(60,dist_prior_event_probs)
x = pd.Series(x,index=dist_prior_event_probs.index)
x.sum()
#x_adj = x*(1/x.sum())
#x_adj.sum()
median_of_dist(dist_prior_event_probs)
median_of_dist(x_adj)
plt.plot(dist_prior_event_probs)
plt.plot(x_adj)




#HAND HELP
i=65 #DECISION INPUT
prob_t_tot(i,dist_prior_event_probs)
1/i
compute_p_t(i,dist_prior_event_probs) #FIXED ACROSS ALL T_tot

#DO MY P(t) add to 1?
catch = ([])
for i in range(0,103):
    catch = np.append(catch,compute_p_t(i,dist_prior_event_probs))

catch.sum()
plt.plot(catch)
#YES

#DOES MY p(t|t_tot) sum to 1
catch = ([])
for i in range(1,103):
    catch = np.append(catch,1/i)
    
catch.sum()
plt.plot(catch)

#EOF