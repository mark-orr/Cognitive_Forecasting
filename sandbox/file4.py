import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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
    #print('t is: ',t)
    vect_t_total = dist.index
    #print('t_total_range is: ',vect_t_total)
    
    catch = np.array([])
    
    for i in vect_t_total:
        #print('t_total is: ',i)
        #prior
        prior = prob_t_tot(i,dist)
        if i < t:
            likelihood = 0
        else: 
            #likelihood = (1/i)*(n_t_tot(i,dist_prior_event_probs,N))
            likelihood = (n_t_tot(i,dist,N)) / ((n_t_tot(i,dist,N))*i)
        p_t = compute_p_t(t,dist)
        likelihood_ratio = likelihood/p_t
        #print('likelihood, p_t, likelihood_ratio, prior, post: ',likelihood, p_t, likelihood_ratio, prior, likelihood_ratio*prior)
        catch = np.append(catch,likelihood_ratio*prior)
        
    return catch



'''DATA FROM SUBJECT 112076'''
df_112076_in = pd.read_csv('Sub_112076_BayesData_r1_r4.csv')
df_112076_tmp = df_112076_in[['user_id','date','0']]
df_112076_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
df_112076_tmp['decision_date'] = df_112076_tmp['decision_date_str'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_112076_tmp['prediction_date'] = df_112076_tmp['prediction_date_str'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_112076 = df_112076_tmp.copy()
df_112076['prediction_duration'] = df_112076.prediction_date - df_112076.decision_date
df_112076['prediction_duration_int'] = df_112076['prediction_duration'].apply(lambda x: x.days)

#WAVE BEGIN TIMES, SO CAN FIX t
begin_w1 = datetime.strptime('20210615', "%Y%m%d")
begin_w2 = datetime.strptime('20211115', "%Y%m%d")
df_112076['begin_w1'] = begin_w1
df_112076['begin_w2'] = begin_w2

df_112076['t_w1'] = df_112076.decision_date - df_112076.begin_w1
df_112076['t_w1_int'] = df_112076['t_w1'].apply(lambda x: x.days)

df_112076['t_w2'] = df_112076.decision_date - df_112076.begin_w2
df_112076['t_w2_int'] = df_112076['t_w2'].apply(lambda x: x.days)

df_112076['prediction_w1'] = df_112076.t_w1_int + df_112076.prediction_duration_int
df_112076['prediction_w2'] = df_112076.t_w2_int + df_112076.prediction_duration_int

df_112076.prediction_duration_int.plot()
df_112076.prediction_w1.plot()
df_112076.prediction_w2.plot()

'''JOB IS TO FIND BEST PRIOR GIVEN THAT T IS NOW FIXED
t_w1_int is var to use AS INPUT TO LIKELIHOOD
prediction_w1 = t_w1_int + prediction_duration_int
'''
#FOR PRIOR
N = 100

'''COMPUTE PRIORS'''
catch_all_prior_over_all_t = []
#COMPUTE PRIORS
for i in range(150,211,5):
    #MAKE PRIOR FOR MEAN as i
    dist_prior = np.random.poisson(i,N)
    dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
    dist_prior_event_probs = dist_prior_event_probs.sort_index()
    
    #GENERATE OVER ALL T AND COLLECT 
    #THEN COLLECT TO LIST OVER PRIORS (OVER i)
    catch_prior_over_all_t = ([])
    for t in range(1,int(dist_prior_event_probs.index[-1])): 
        dist_1 = compute_posterior(t, dist_prior_event_probs)
        dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
        catch_prior_over_all_t = np.append(catch_prior_over_all_t, median_of_dist(dist_1))
    catch_all_prior_over_all_t.append(catch_prior_over_all_t)

    
    
'''****TEST FOR WAVE 1****DELTA'''
'''HAVE PRIORS NOW COMPUTE ERROR
ASSUME ONE PRIOR FOR ALL JUDGEMENTS OF S'''
catch_all_t_over_t_over_p = []
catch_all_optimal_pred_over_t_over_p = []
catch_all_human_pred_over_t_over_p = []
catch_all_error_over_t_over_p = []

for j in catch_all_prior_over_all_t: #LOOP OVER PRIORS
    
    print('NEW PRIOR')
    catch_all_t_over_t = ([])
    catch_all_optimal_pred_over_t = ([])
    catch_all_human_pred_over_t = ([])
    catch_all_error_over_t = ([])

    for i in range(0,79):#CAPTURE HUMAN DATA T AND PRED
        t = df_112076.t_w1_int.iloc[i] #P
        print('t',t)
        catch_all_t_over_t = np.append(catch_all_t_over_t,t)
        #optimal_pred = catch_all_prior_over_all_t[0][t-1]#index zero is t=1
        #print('LEN of J',len(j))
        if t-1<len(j):
            optimal_pred = j[t-1]#index zero is t=1
        else:
            print('T-1 OUT OF RANGE OF Ts in PRIOR')
            optimal_pred = j[-1]
        catch_all_optimal_pred_over_t = np.append(catch_all_optimal_pred_over_t,optimal_pred)
        print('optimal pred',optimal_pred)
        human_pred = df_112076.prediction_w1.iloc[i]
        catch_all_human_pred_over_t = np.append(catch_all_human_pred_over_t,human_pred)
        print('human pred', human_pred)
        error = human_pred - optimal_pred
        print('error',error)
        catch_all_error_over_t = np.append(catch_all_error_over_t,error)
    
    catch_all_t_over_t_over_p.append(catch_all_t_over_t)
    catch_all_optimal_pred_over_t_over_p.append(catch_all_optimal_pred_over_t)
    catch_all_human_pred_over_t_over_p.append(catch_all_human_pred_over_t)
    catch_all_error_over_t_over_p.append(catch_all_error_over_t)

'''PICK BEST PRIOR'''
for i in catch_all_error_over_t_over_p: plt.plot(i)
#SAVE FOR LATER 
error_w1 = catch_all_error_over_t_over_p.copy()
'''THE PLOT ABOVE REALLY NAILS THE TRANSITION
1. THIS IS A DECISION PLOT, ALL DECISIONS
2. USE IT TO DECIDE WHERE TO CUT PRIORS.
3. THEN RUN SUBSET FOR EACH PRIOR AND COMPUTE
BEST PRIOR BY ERROR
'''
#THESE ARE GOOD TOO
len(catch_all_optimal_pred_over_t_over_p)
y = catch_all_optimal_pred_over_t_over_p[12]
x = catch_all_human_pred_over_t_over_p[12]
plt.scatter(x,y)

plt.plot(df_112076.prediction_duration_int)
#BY EYE TEST, PRIOR SHIFT OCCURS AT
min_index_neg_pred_duration = min(df_112076.prediction_duration_int.loc[df_112076.prediction_duration_int < 0].index)

df_112076_w1 = df_112076.iloc[0:22]
'''RUN IT AGAIN WITH TRUNCATED DATA'''
    
'''****TEST FOR WAVE 1****DELTA'''
'''HAVE PRIORS NOW COMPUTE ERROR
ASSUME ONE PRIOR FOR ALL JUDGEMENTS OF S'''
catch_all_t_over_t_over_p = []
catch_all_optimal_pred_over_t_over_p = []
catch_all_human_pred_over_t_over_p = []
catch_all_error_over_t_over_p = []

for j in catch_all_prior_over_all_t: #LOOP OVER PRIORS
    
    print('NEW PRIOR')
    catch_all_t_over_t = ([])
    catch_all_optimal_pred_over_t = ([])
    catch_all_human_pred_over_t = ([])
    catch_all_error_over_t = ([])

    for i in range(0,21):#CAPTURE HUMAN DATA T AND PRED
        t = df_112076_w1.t_w1_int.iloc[i] #P
        print('t',t)
        catch_all_t_over_t = np.append(catch_all_t_over_t,t)
        #optimal_pred = catch_all_prior_over_all_t[0][t-1]#index zero is t=1
        #print('LEN of J',len(j))
        if t-1<len(j):
            optimal_pred = j[t-1]#index zero is t=1
        else:
            print('T-1 OUT OF RANGE OF Ts in PRIOR')
            optimal_pred = j[-1]
        catch_all_optimal_pred_over_t = np.append(catch_all_optimal_pred_over_t,optimal_pred)
        print('optimal pred',optimal_pred)
        human_pred = df_112076_w1.prediction_w1.iloc[i]
        catch_all_human_pred_over_t = np.append(catch_all_human_pred_over_t,human_pred)
        print('human pred', human_pred)
        error = human_pred - optimal_pred
        print('error',error)
        catch_all_error_over_t = np.append(catch_all_error_over_t,error)
    
    catch_all_t_over_t_over_p.append(catch_all_t_over_t)
    catch_all_optimal_pred_over_t_over_p.append(catch_all_optimal_pred_over_t)
    catch_all_human_pred_over_t_over_p.append(catch_all_human_pred_over_t)
    catch_all_error_over_t_over_p.append(catch_all_error_over_t)

'''PICK BEST PRIOR'''
for i in catch_all_error_over_t_over_p: plt.plot(i)
for i in catch_all_error_over_t_over_p:
    print(i, np.mean(i), np.std(i))
for i in range(150,211,5):
    print(i)
'''MIN ERROR GETS IT, GOES 170 or 175'''
plot_matter = np.random.poisson(175,1000)
plt.hist(plot_matter,bins=100)


'''****TEST FOR WAVE 2****OMICRON'''
'''HAVE PRIORS NOW COMPUTE ERROR
ASSUME ONE PRIOR FOR ALL JUDGEMENTS OF S'''
catch_all_t_over_t_over_p = []
catch_all_optimal_pred_over_t_over_p = []
catch_all_human_pred_over_t_over_p = []
catch_all_error_over_t_over_p = []

for j in catch_all_prior_over_all_t: #LOOP OVER PRIORS
    
    print('NEW PRIOR')
    catch_all_t_over_t = ([])
    catch_all_optimal_pred_over_t = ([])
    catch_all_human_pred_over_t = ([])
    catch_all_error_over_t = ([])

    for i in range(0,79):#CAPTURE HUMAN DATA T AND PRED
        t = df_112076.t_w2_int.iloc[i] #P
        print('t',t)
        catch_all_t_over_t = np.append(catch_all_t_over_t,t)
        #optimal_pred = catch_all_prior_over_all_t[0][t-1]#index zero is t=1
        #print('LEN of J',len(j))
        if t-1<len(j):
            optimal_pred = j[t-1]#index zero is t=1
        else:
            print('T-1 OUT OF RANGE OF Ts in PRIOR')
            optimal_pred = j[-1]
        catch_all_optimal_pred_over_t = np.append(catch_all_optimal_pred_over_t,optimal_pred)
        print('optimal pred',optimal_pred)
        human_pred = df_112076.prediction_w2.iloc[i]
        catch_all_human_pred_over_t = np.append(catch_all_human_pred_over_t,human_pred)
        print('human pred', human_pred)
        error = human_pred - optimal_pred
        print('error',error)
        catch_all_error_over_t = np.append(catch_all_error_over_t,error)
    
    catch_all_t_over_t_over_p.append(catch_all_t_over_t)
    catch_all_optimal_pred_over_t_over_p.append(catch_all_optimal_pred_over_t)
    catch_all_human_pred_over_t_over_p.append(catch_all_human_pred_over_t)
    catch_all_error_over_t_over_p.append(catch_all_error_over_t)


'''PICK BEST PRIOR'''
for i in catch_all_error_over_t_over_p: plt.plot(i)
error_w2 = catch_all_error_over_t_over_p.copy()
'''THE PLOT ABOVE REALLY NAILS THE TRANSITION
1. THIS IS A DECISION PLOT, ALL DECISIONS
'''

'''PLOT BOTH TOGETHER'''
'''THIS PLOT IS A SWITCH IN PRIORS'''
for i in error_w1: plt.plot(i)
for i in error_w2: plt.plot(i)


#THESE TWO ARE INPUTS TO ERROR CALC.

    #HAVE A PRIOR NOW RUN ERRORS
    for d in range(0,len(df_112076)):
        
        compute_posterior(df_112076.)
        
        catch_bayes = ([])
        catch_error_by_prior = ([])
        catch_predicted_duration = ([])
        for t in range(1,int(dist_prior_event_probs.index[-1])): 
            dist_1 = compute_posterior(t, dist_prior_event_probs)
            dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
            catch_bayes = np.append(catch_bayes, median_of_dist(dist_1))
            pred_dur = t + df_112076.pred_duration_int.iloc[d]
            error = pred_dur - median_of_dist(dist_1)
            catch_error_by_prior = np.append(catch_error_by_prior,error)
            catch_predicted_duration = np.append(catch_predicted_duration,pred_dur)

for d in range(0,len(df_112076):
    #
    
    catch_error_zero_ts_for_decision = [] #VECTOR OF T OF MIN ERROR FOR EACH PRIOR
    
    for i in range(110,111):
        dist_prior = np.random.poisson(i,N)

    #COMPUTE PROBS FOR EACH EVENT
        dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
        dist_prior_event_probs = dist_prior_event_probs.sort_index()

    #GET OPTIMAL OVER T
        catch_bayes = ([])
        catch_error_by_prior = ([])
        catch_predicted_duration = ([])
        for t in range(1,int(dist_prior_event_probs.index[-1])): 
            dist_1 = compute_posterior(t, dist_prior_event_probs)
            dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
            catch_bayes = np.append(catch_bayes, median_of_dist(dist_1))
            pred_dur = t + df_112076.pred_duration_int.iloc[d]
            error = pred_dur - median_of_dist(dist_1)
            catch_error_by_prior = np.append(catch_error_by_prior,error)
            catch_predicted_duration = np.append(catch_predicted_duration,pred_dur)
    
        catch_ts.append(catch_bayes)
        catch_error.append(catch_error_by_prior)
        catch_pred_dur.append(catch_predicted_duration)
        
        catch_error_zero_ts_tmp = [j for j, value in enumerate(catch_error_by_prior) if value == 0]
        if len(catch_error_zero_ts_tmp)>1:
                catch_error_zero_ts_tmp = [catch_error_zero_ts_tmp[0]]
        
        catch_error_zero_ts_for_decision.append(catch_error_zero_ts_tmp)
    
    catch_error_zero_ts_across_decisions.append(catch_error_zero_ts_for_decision)
    
     
        '''END LOOP HERE '''



'''OLD'''

'''GENERATE OPTIMALl, GIVEN T AND PRIOR, AND ONE Ss PREDICTION'''
catch_ts = [] #OPTIMAL DECISION FROM POSTERIOR
catch_error = [] #ERROR ACROSS T FOR PRIORS
catch_pred_dur = [] #PRED DUR ACROSS T FOR PRIORS
catch_error_zero_ts_across_decisions = [] #T OF MIN ERROR FOR EACH PRIOR for each DECISION

#for d in [0,1]:
for d in range(0,len(df_112076.pred_duration_int)):
    
    catch_error_zero_ts_for_decision = [] #VECTOR OF T OF MIN ERROR FOR EACH PRIOR
    
    for i in range(110,160,10):
        dist_prior = np.random.poisson(i,N)

    #COMPUTE PROBS FOR EACH EVENT
        dist_prior_event_probs = pd.Series(dist_prior).value_counts()/pd.Series(dist_prior).value_counts().sum()
        dist_prior_event_probs = dist_prior_event_probs.sort_index()

    #GET OPTIMAL OVER T
        catch_bayes = ([])
        catch_error_by_prior = ([])
        catch_predicted_duration = ([])
        for t in range(1,int(dist_prior_event_probs.index[-1])): 
            dist_1 = compute_posterior(t, dist_prior_event_probs)
            dist_1 = pd.Series(dist_1, index=dist_prior_event_probs.index)
            catch_bayes = np.append(catch_bayes, median_of_dist(dist_1))
            pred_dur = t + df_112076.pred_duration_int.iloc[d]
            error = pred_dur - median_of_dist(dist_1)
            catch_error_by_prior = np.append(catch_error_by_prior,error)
            catch_predicted_duration = np.append(catch_predicted_duration,pred_dur)
    
        catch_ts.append(catch_bayes)
        catch_error.append(catch_error_by_prior)
        catch_pred_dur.append(catch_predicted_duration)
        
        catch_error_zero_ts_tmp = [j for j, value in enumerate(catch_error_by_prior) if value == 0]
        if len(catch_error_zero_ts_tmp)>1:
                catch_error_zero_ts_tmp = [catch_error_zero_ts_tmp[0]]
        
        catch_error_zero_ts_for_decision.append(catch_error_zero_ts_tmp)
    
    catch_error_zero_ts_across_decisions.append(catch_error_zero_ts_for_decision)
    
     
        '''END LOOP HERE '''

'''GIVEN PREDICTED DURATION OF X, GET MIN ERROR T FOR EACH PRIORS OPTIMAL'''

for i in range(0,len(catch_error_zero_ts)):
    print(catch_error_zero_ts[i][0])
    print(catch_ts[i][catch_error_zero_ts[i][0]])
    print(catch_pred_dur[i][catch_error_zero_ts[i][0]])
    print(catch_error[i][catch_error_zero_ts[i][0]])

#COMPARE
catch_error_zero_ts_across_decisions
np.array(catch_error_zero_ts_across_decisions)[:,0,0]
df_112076.pred_duration_int

plt.plot(np.array(catch_error_zero_ts_across_decisions)[:,0,0])
plt.plot(np.array(catch_error_zero_ts_across_decisions)[:,1,0])
plt.plot(np.array(catch_error_zero_ts_across_decisions)[:,2,0])
plt.plot(np.array(catch_error_zero_ts_across_decisions)[:,3,0])
plt.plot(np.array(catch_error_zero_ts_across_decisions)[:,4,0])
plt.plot(df_112076.pred_duration_int)




'''DEV AND TESTING'''
    
'''SO FOR EACH PRIOR AND t WE COMPUTE (t + pred_duration_int) - catch_ts '''
poisson_mean_list = [i for i in range(80,160)]   



catch_error_zero_ts_for_plot = ([])
for i in range(0,len(catch_ts)):
    catch_error_zero_ts = [j for j, value in enumerate(catch_error[i]) if value == 0]
    if len(catch_error_zero_ts)>1:
        catch_error_zero_ts = [catch_error_zero_ts[0]]
    print(catch_error_zero_ts[0])
    catch_error_zero_ts_for_plot = np.append(catch_error_zero_ts_for_plot,catch_error_zero_ts[0])
    

    
for i in catch_ts: plt.plot(i)
for i in range(0,50): 
    t = int(catch_error_zero_ts_for_plot[i])
    pred = catch_pred_dur[i][t]
    print(t,pred)
    plt.plot(t,pred,'ro')

    
    
'''TESTING MAIN LOOP'''  
catch_error_zero_ts_for_plot = ([])
for i in range(0,len(catch_ts)):
    catch_error_zero_ts = [j for j, value in enumerate(catch_error[i]) if value == 0]
    print(catch_error_zero_ts[0])
    
    if len(catch_error_zero_ts)>1:
        catch_error_zero_ts = [catch_error_zero_ts[0]]
    print(catch_error_zero_ts[0])
    #catch_error_zero_ts_for_plot = np.append(catch_error_zero_ts_for_plot,catch_error_zero_ts[0])
        
        
        
        


'''FROM FILE2'''
'''POINTS FOR PLOT FOR SUBJECT'''
#FOR SUBJE 112076
catch_sub = []
offset = 0
for i in df_112076.subj_input_int:
    #print(i)
    j = i - offset
    x = compute_posterior(j, dist_prior_event_probs)
    x = pd.Series(x,index=dist_prior_event_probs.index)
    x.sum()
    #print(median_of_dist(x))
    catch_sub.append([j,median_of_dist(x)])


'''PLOT OVER t'''
plt.plot(catch,label='median')
plt.legend()
plt.title('DeltaWave_Prior')
plt.xlabel('Subjective Days into Epidemic')
plt.ylabel('Decision Value from Posterior')
plt.axvline(x=median_of_dist(dist_prior_event_probs), c='b',dashes=(2,2,2,2),linewidth=1)
plt.axhline(y=catch[median_of_dist(dist_prior_event_probs)], c='b',dashes=(2,2,2,2),linewidth=1)
#for i in range(0,len(catch_4)): 
#    plt.plot(catch_4[i][0],catch_4[i][1],'ro')
for i in range(0,len(catch_sub)): 
    plt.plot(catch_sub[i][0],(df_112076.decision_output_int[i]-offset),'ro')

#plt.savefig('Tmp.png',dpi=200)
#JUST A TEST THAT CPATURING THE HUMAN OUTPUT
for i in range(0,len(catch_sub)):
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

