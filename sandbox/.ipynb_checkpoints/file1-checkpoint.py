import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''DEF'''
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
#prob_t_tot(34,dist_prior_event_probs)

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
#n_t_tot(70,dist_prior_event_probs,N)

def median_of_dist(dist):
    '''
    dist is a pd.Series
    returns the median index value which should be
    the value of the poisson
    '''
    return dist.loc[dist.cumsum()>0.5].index[0]
#EG
#median_of_dist(dist_prior_event_probs)

def median_of_dist_p(dist,p):
    '''
    dist is a pd.Series
    returns the median index value which should be
    the value of the poisson
    p = prob value to grab above, 0.5 is median
    '''
    return dist.loc[dist.cumsum()>p].index[0]
#EG
#median_of_dist(dist_prior_event_probs)




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
#compute_p_t(5,dist_prior_event_probs)
#compute_p_t(75,dist_prior_event_probs)
'''compute_p_t LOGIpC'''
#dist_prior_event_probs.index
#one_over_ttot = pd.Series(1/dist_prior_event_probs.index,index=dist_prior_event_probs.index)



'''EXAMPLE LOGIC
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
        if i < t:
            likelihood = 0
        else: 
            #likelihood = (1/i)*(n_t_tot(i,dist_prior_event_probs,N))
            likelihood = (n_t_tot(i,dist,N)) / ((n_t_tot(i,dist,N))*i)
        p_t = compute_p_t(t,dist)
        likelihood_ratio = likelihood/p_t
        print('likelihood, p_t, likelihood_ratio, prior, post: ',likelihood, p_t, likelihood_ratio, prior, likelihood_ratio*prior)
        catch = np.append(catch,likelihood_ratio*prior)
        
    return catch



'''METHOD TESTS'''
'''STEPS
1. Make Prior Distribution
2. compute t_tot over range of t
3. compute decision/estimate from #2 for all t
3. plot results over t'''

'''GENERATE PRIOR'''
#GENERATOR
N = 10000
#dist_prior = np.random.poisson(120,N)
dist_prior = np.random.exponential(20,N)
dist_prior = np.round(dist_prior).copy()
print(dist_prior)
plt.hist(dist_prior,bins=30)
#plt.title('Prior Distribution (Poisson, M120)')
#plt.xlabel('Duration of Epidemic')
#plt.ylabel('Freq')
#plt.savefig('Prior_Dist_Example_Poisson_M120.png',dpi=200)
plt.title('Prior Distribution (Exponential, M20)')
plt.xlabel('Duration of Epidemic')
plt.ylabel('Freq')
plt.savefig('Prior_Dist_Example_Exponential_M20.png',dpi=200)

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


#RUN OVER RANGE OF t
catch = ([])
catch_2 = ([])
catch_3 = ([])
for t in range(1,int(dist_prior_event_probs.index[-1])): 
    dist_1 = compute_posterior(t, dist_prior_event_probs)
    dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
    print('dist_1 sum: ', dist_1.sum())
    print('median_prior', median_of_dist(dist_prior_event_probs))
    print('median_posterior', median_of_dist(dist_1))
    print('high_posterior', median_of_dist_p(dist_1,0.9))
    print('low_posterior', median_of_dist_p(dist_1,0.1))
    catch = np.append(catch, median_of_dist(dist_1))
    catch_2 = np.append(catch_2, median_of_dist_p(dist_1,0.9))
    catch_3 = np.append(catch_3, median_of_dist_p(dist_1,0.1))



    
    
'''PLOT POISSON AS EXAMPLE'''
mean_of_dist = 120
plt.plot(catch,label='median')
plt.plot(catch_2,label='high threshold')
plt.plot(catch_3,label='low threshold')
plt.legend()
plt.title('Poisson_Mean_120_Days_N10')
plt.xlabel('Subjective Days into Epidemic')
plt.ylabel('Decision Value from Posterior')
plt.axvline(x=mean_of_dist, c='b',dashes=(2,2,2,2),linewidth=1)
plt.axhline(y=catch[mean_of_dist], c='b',dashes=(2,2,2,2),linewidth=1)

plt.savefig('Poisson_Mean_120D_N10.png',dpi=200)

    
'''PLOT EXPONENTIAL AS EXAMPLE'''
mean_of_dist = 20
plt.plot(catch,label='median')
plt.plot(catch_2,label='high threshold')
plt.plot(catch_3,label='low threshold')
plt.legend()
plt.title('Exponential_Mean_20_Days_N10000')
plt.xlabel('Subjective Days into Epidemic')
plt.ylabel('Decision Value from Posterior')
plt.axvline(x=mean_of_dist, c='b',dashes=(2,2,2,2),linewidth=1)
plt.axhline(y=catch[mean_of_dist], c='b',dashes=(2,2,2,2),linewidth=1)

plt.savefig('Exponential_Mean_20D_N10000.png',dpi=200)
'''END MAIN'''


'''HOW IT WORKS'''
#RUN FOR ONE t
x = compute_posterior(40,dist_prior_event_probs)
x = pd.Series(x,index=dist_prior_event_probs.index)
x.sum()
#x_adj = x*(1/x.sum())
#x_adj.sum()
prior_med = median_of_dist(dist_prior_event_probs)
decision_med = median_of_dist(x)
plt.title('Decision Process with 40 Days as Input')
plt.plot(dist_prior_event_probs,label=f'prior, median={prior_med}')
plt.plot(x,label=f'decision, median={decision_med}')
plt.axvline(x=prior_med, c='b',dashes=(2,2,2,2),linewidth=1)
plt.axvline(x=decision_med, c='b',linewidth=1)
plt.legend()

plt.savefig('Example_Decision_Dist_Poisson_Mean_120_N1000_Input40.png',dpi=200)


'''HELPER'''
'''RUN ONE VALUE OF t'''

#RUN FOR ONE t
x = compute_posterior(130,dist_prior_event_probs)
x = pd.Series(x,index=dist_prior_event_probs.index)
x.sum()
#x_adj = x*(1/x.sum())
#x_adj.sum()
median_of_dist(dist_prior_event_probs)
median_of_dist(x)
plt.plot(dist_prior_event_probs)
plt.plot(x)










'''SCRAP AND TESTING MESS BELOW'''
#HAND HELP
i=60 #DECISION INPUT
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


'''NOT CURRENTLY RELVANT
THIS JUST FORECED i = + 1 BECAUSE OF DIV BY ZERO, BUT NOT NECESSARY

'''
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


#EOF