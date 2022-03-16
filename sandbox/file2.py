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



'''DATA FROM SUBJECT 112076'''
df_112076 = pd.read_csv('Sub_112076_BayesData.csv')
#FOR DEMO, ASSUME 7-15-2021 is start of epidemic

'''DATA FROM VIRGINIA'''
df_va_ts = pd.read_csv('VA.timeseries.csv')

df_ts = df_va_ts[['date','fips','actuals.newCases']]
df_ts_delta = df_ts['actuals.newCases'][499:670]#MAKES SERIES GIVEN DATES OF ASSUMED start of epidemic to 
#11-14-2021
df_ts.date[670]#DATE
S_ts = df_ts_delta.reset_index(drop=True)#ALREADY SORTED
dist_prior_event_probs = S_ts/S_ts.sum()
N = S_ts.sum()
#NEED TO DROP ZEROs
dist_prior_event_probs = dist_prior_event_probs.loc[dist_prior_event_probs>0]
plt.plot(dist_prior_event_probs)
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

'''POINTS FOR PLOT FOR SUBJECT'''
#FOR SUBJE 112076
catch_4 = []
for i in df_112076.subj_input_int:
    print(i)
    x = compute_posterior(i, dist_prior_event_probs)
    x = pd.Series(x,index=dist_prior_event_probs.index)
    x.sum()
    print(median_of_dist(x))
    catch_4.append([i,median_of_dist(x)])


'''PLOT OVER t'''
plt.plot(catch,label='median')
plt.plot(catch_2,label='high threshold')
plt.plot(catch_3,label='low threshold')
plt.legend()
plt.title('DeltaWave')
plt.xlabel('Subjective Days into Epidemic')
plt.ylabel('Decision Value from Posterior')
plt.axvline(x=median_of_dist(dist_prior_event_probs), c='b',dashes=(2,2,2,2),linewidth=1)
plt.axhline(y=catch[median_of_dist(dist_prior_event_probs)], c='b',dashes=(2,2,2,2),linewidth=1)
#for i in range(0,len(catch_4)): 
#    plt.plot(catch_4[i][0],catch_4[i][1],'ro')
for i in range(0,len(catch_4)): 
    plt.plot(catch_4[i][0],df_112076.decision_output_int[i],'ro')

plt.savefig('DeltaWave_Decision_Predictions.png',dpi=200)
#JUST A TEST THAT CPATURING THE HUMAN OUTPUT
for i in range(0,len(catch_4)):
    print(df_112076.decision_output_int[i])


    
    

'''SCRAP AND POSSIBLE HELPERS'''
#FOR INDIVIDUAL USE
x = compute_posterior(130, dist_prior_event_probs)
x = pd.Series(x,index=dist_prior_event_probs.index)
x.sum()
median_of_dist(x)

'''NEED THIS TRICK'''
tmp = dist_prior_event_probs.loc[120]
dist_prior_event_probs.loc[120] = tmp/2
dist_prior_event_probs = dist_prior_event_probs.append(pd.Series([tmp/2], index=[145])) 
    
'''NOTES
1. Might try artificially extending the delta dist 
to capture idea that in their minds, another wave was not 
coming yet (just cut final prob in 1/2 and put way out)
2. Next Step:  Will improve the data structure so has integer subj_input and decision output for direct 
contact with code above.  Will make predictions based on that...
Can also make 
'''

