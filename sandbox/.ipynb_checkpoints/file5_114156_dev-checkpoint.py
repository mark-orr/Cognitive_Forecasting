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

S_no = 114156

'''DATA FROM SUBJECT 114156'''
'''GENERATE BASIC DATA STRUCTURE'''
df_114156_in = pd.read_csv(f'Sub_{S_no}_BayesData_r1_r4.csv')
df_114156_tmp = df_114156_in[['user_id','date','0']]
df_114156_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
df_114156_tmp['decision_date'] = df_114156_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_114156_tmp['prediction_date'] = df_114156_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_114156 = df_114156_tmp.copy()
df_114156['prediction_duration'] = df_114156.prediction_date - df_114156.decision_date
df_114156['prediction_duration_int'] = df_114156['prediction_duration'].map(lambda x: x.days)
'''ESTIMATE OF t_0 for omicron = 1-19-22'''
'''JUST A NOTE'''

#WAVE BEGIN TIMES, SO CAN FIX t
begin_w1 = datetime.strptime('20210623', "%Y%m%d")
begin_w2 = datetime.strptime('20211129', "%Y%m%d")
df_114156['begin_w1'] = begin_w1
df_114156['begin_w2'] = begin_w2
'''ESTIMATE OF PRIOR FROM W1'''
begin_w1 - begin_w2
#159 days BUT THIS IS FIXED ON A FIXED ACROSS S PRIOR

df_114156['t_w2'] = df_114156.decision_date - df_114156.begin_w2
df_114156['t_w2_int'] = df_114156['t_w2'].map(lambda x: x.days)

df_114156['prediction_w2'] = df_114156.t_w2_int + df_114156.prediction_duration_int
#NEXT SHOULD BE EQUVALENT TO ABOVE
df_114156['prediction_w2_test'] = df_114156.prediction_date - df_114156.begin_w2
#TEST GOOD

#STOPPED TESTING HERE

'''THESE TWO ARE TELLING 
FIRST SHOWS PREDICTION
SECOND SHOWS PREDICTED DATE
Good For Publication, MAYBE
'''
df_114156.prediction_duration_int.plot()
df_114156.prediction_w2.plot()

'''FURTHER EXPORE OF MIN JUDGMENTS
1. WERE THEY ALL MINERS
2. How INTERPRET NEGATIVES FROM '''
df_114156[df_114156.prediction_duration_int<0]
df_114156[['decision_date','prediction_duration_int']]
'''THESE ARE ALL MINER JUDGEMENTS'''

'''
ONLY INCLUDE DECISION DATES AFTER THE FOLLOWING CRITERIA
1. ON OR AFTER ESTIMATED t_0 DATE
2. May DELETE OR RE-INTERPRET NEGATIVEs 
'''
tmp_t = pd.to_datetime(datetime.strptime('2021-12-25T17:00:00','%Y-%m-%dT%H:%M:%S')) 
df_114156_w2 = df_114156[df_114156.decision_date > tmp_t ]


'''JOB IS TO FIND BEST PRIOR GIVEN THAT T IS NOW FIXED
t_w1_int is var to use AS INPUT TO LIKELIHOOD
prediction_w1 = t_w1_int + prediction_duration_int
'''
#FOR PRIOR
N = 10

'''COMPUTE PRIORS'''
catch_all_prior_over_all_t = []
catch_prior_index = []
#COMPUTE PRIORS
for i in range(80,131,5):
    catch_prior_index.append(i)
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

'''****TEST FOR WAVE 2****OMICRON'''
'''HAVE PRIORS NOW COMPUTE ERROR
ASSUME ONE PRIOR FOR ALL JUDGEMENTS OF S'''
catch_all_t_over_t_over_p = []
catch_all_optimal_pred_over_t_over_p = []
catch_all_human_pred_over_t_over_p = []
catch_all_error_over_t_over_p = []
catch_all_date_over_t_over_p = []

for j in catch_all_prior_over_all_t: #LOOP OVER PRIORS
    
    print('NEW PRIOR')
    print('NEW PRIOR')
    catch_all_t_over_t = ([])
    catch_all_optimal_pred_over_t = ([])
    catch_all_human_pred_over_t = ([])
    catch_all_error_over_t = ([])
    catch_all_date_over_t = ([])

    for i in range(0,len(df_114156_w2)):#CAPTURE HUMAN DATA T AND PRED
        t = df_114156_w2.t_w2_int.iloc[i] #P
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
        human_pred = df_114156_w2.prediction_w2.iloc[i]
        catch_all_human_pred_over_t = np.append(catch_all_human_pred_over_t,human_pred)
        print('human pred', human_pred)
        error = human_pred - optimal_pred
        print('error',error)
        catch_all_error_over_t = np.append(catch_all_error_over_t,error)
        catch_all_date_over_t = np.append(catch_all_date_over_t,df_114156_w2.decision_date.iloc[i])
        print('decision_date',df_114156_w2.decision_date.iloc[i])
    
    catch_all_t_over_t_over_p.append(catch_all_t_over_t)
    catch_all_optimal_pred_over_t_over_p.append(catch_all_optimal_pred_over_t)
    catch_all_human_pred_over_t_over_p.append(catch_all_human_pred_over_t)
    catch_all_error_over_t_over_p.append(catch_all_error_over_t)
    catch_all_date_over_t_over_p.append(catch_all_date_over_t)


'''PICK BEST PRIOR'''
for i in catch_all_error_over_t_over_p: plt.plot(i)
#S HAS TWO REGIMES (one with constant decreasing)
#SEE:
df_error = pd.DataFrame(catch_all_error_over_t_over_p).T
#df_error['decision_date'] = df_114156_w2.decision_date.reset_index(drop=True)
df_error.index = df_114156_w2.decision_date.reset_index(drop=True)
df_error.columns = catch_prior_index
df_error.abs().idxmin(axis=1).plot()
df_error.abs().idxmin(axis=1)
df_error.abs().min(axis=1)
df_error.min(axis=both)


df_error.min(axis=1)


#COMPUTE ERROR SEP FOR FIRST AND SECOND REGIME
#REGIME 1
for i in catch_all_error_over_t_over_p:
    print(np.mean(i[0:17]), np.std(i[0:17]))
#REGIME 2
for i in catch_all_error_over_t_over_p:
    print(np.mean(i[17:]), np.std(i[17:]))
#MAP TO THE PRIORS
for i in range(30,51,5): print(i)
'''BEST PRIOR is 90 for first regime and 110 for sec.'''

'''GOOD PLOT FOR FOLKS'''
plot_matter = np.random.poisson(90,100)
plt.hist(plot_matter,bins=100,label='90')
plot_matter = np.random.poisson(110,100)
plt.hist(plot_matter,bins=100,label='110')
plt.legend()

'''PLAYING WITH PLOTS'''
plt.scatter(catch_all_optimal_pred_over_t_over_p[0],catch_all_human_pred_over_t_over_p[0])
    
catch_all_error_over_t_over_p
    
    
pd.to_datetime(datetime.strptime('2021-12-25T17:00:00','%Y-%m-%dT%H:%M:%S'))
df_error.iat[0,3]

#PAYDIRT
plot_this = df_error.abs().idxmin(axis=1)
#IF HAVE MUTIPLE RETURNS ON THIS
df_error.at[plot_this.index[2],plot_this.iloc[2]]
type(df_error.at[plot_this.index[1],plot_this.iloc[1]])

isinstance(1, np.floating)

#ELSE< SIMPLE CASE
type(df_error.at[plot_this.index[2],plot_this.iloc[2]])
for j in range(0,len(plot_this)):
    print('j ',j)
    if isinstance(df_error.at[plot_this.index[j],plot_this.iloc[j]],np.floating):
        print('i: is float', i)
        print(df_error.at[plot_this.index[j],plot_this.iloc[j]])
    else:
        print('i: is len > 1', i)
        print(len(df_error.at[plot_this.index[j],plot_this.iloc[j]]))
        for i in range(0,len(df_error.at[plot_this.index[j],plot_this.iloc[j]])):
            print(df_error.at[plot_this.index[j],plot_this.iloc[j]][i])


for i in range(0,len(plot_this)):
    print(df_error.at[plot_this.index[i],plot_this.iloc[i]])
df

'''
ABOVE IS COOL, BUT NEED SIMPLE ANSWER
CAN USE ABOVE TO GET DATE AND PRIOR INFO IN COMPACT FORMAT AND BELOW FOR PLOTTING
'''
df_error_2 = df_error.reset_index(drop=True).copy()
catch_df_error_min = []
plot_this_2 = df_error_2.abs().idxmin(axis=1)
for i in df_error_2.index:
    tmp = df_error_2.at[i,plot_this_2.iloc[i]]
    print(tmp)
    catch_df_error_min.append(tmp)
S_for_bar = pd.Series(catch_df_error_min,index=plot_this.index)

#BAR PLOT WITH ANNOTATIONS FOR PRIOR, DOESN"T RELLAY WORK
fig, ax = plt.subplots()

bar_x = S_for_bar.index
bar_height = S_for_bar
bar_tick_label = S_for_bar.index
bar_label = plot_this_2

bar_plot = plt.bar(bar_x,bar_height,tick_label=bar_tick_label)

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)

autolabel(bar_plot)
plt.xticks(rotation=90)
plt.ylim(-4,4)
plt.axhline(y=0, c='b',dashes=(2,2,2,2),linewidth=1)
plt.title('Add text for each bar with matplotlib')

plt.savefig("add_text_bar_matplotlib_01.png", bbox_inches='tight')
plt.show()


#JUST NEED TO ADD THE PRIORS


#OR THIS:
for i in plot_this:
    plot_matter = np.random.poisson(i,100)
    plt.hist(plot_matter,bins=100,label=i)
#MESSY
range_of_priors = [plot_this.min(),plot_this.max()]
for i in range_of_priors:
    plot_matter = np.random.poisson(i,1000)
    plt.hist(plot_matter,bins=20,label=i)
plt.legend()
    
plot_matter = np.random.poisson(110,100)
plt.hist(plot_matter,bins=100,label='110')
plt.legend()

###THIS WORKS


'''THIS IS AN INSTRUCTURE DATA STRUCTURE'''
df_114156[['t_w2_int','prediction_w2','prediction_duration_int','decision_date','prediction_date']]

#EOF
#EOF
#EOF
