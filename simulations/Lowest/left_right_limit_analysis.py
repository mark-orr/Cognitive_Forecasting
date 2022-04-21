import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack_testing')
import bayes

datadir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/InputData'

'''WILL DO:
RD 3 and 4 for lowest bc judgements were left-mins mostly
RD 1, 2 and 3 for highest because peak was first day of 
'''

'''DATES FOR MIN MAX'''
min_r1 = datetime.strptime('20211111', "%Y%m%d")
max_r1 = datetime.strptime('20220202', "%Y%m%d")#offset of -1 from published date accounted for here
min_r2 = datetime.strptime('20211203', "%Y%m%d")
max_r2 = datetime.strptime('20220224', "%Y%m%d")#offset of -1 from published date accounted for here
min_r3 = datetime.strptime('20211224', "%Y%m%d")
max_r3 = datetime.strptime('20220317', "%Y%m%d")#offset of -1 from published date accounted for here
min_r4 = datetime.strptime('20220114', "%Y%m%d")
max_r4 = datetime.strptime('20220407', "%Y%m%d")#offset of -1 from published date accounted for here



'''QUESTION'''
'''TEST FOR question level data.'''
df_S_in = pd.read_csv(f'{datadir}/AllSubs_q8568.csv')
df_S_tmp = df_S_in[['user_id','date','0']]
df_S_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_use = df_S_tmp.copy()
df_use = df_use.sort_values(by='decision_date')

#RD 1 Q
df_use['min_r1'] = min_r1
df_use['min_diff_r1'] = df_use['min_r1'] - df_use['prediction_date'] 
df_use['min_diff_r1_days'] = df_use.min_diff_r1.map(lambda x: x.days)
df_use['max_r1'] = max_r1
df_use['max_diff_r1'] = df_use['prediction_date'] - df_use['max_r1']
df_use['max_diff_r1_days'] = df_use.max_diff_r1.map(lambda x: x.days)
'''NEED SIMPLE METHOD TO DETECT MAX-MINers'''

'''MAXERS'''
df_use['maxers'] = 0
df_use.loc[df_use.max_diff_r1_days==0,'maxers'] = 1
df_use.maxers.value_counts()

df_use['max_gooders'] = 0
df_use.loc[df_use.max_diff_r1_days<0,'max_gooders'] = 1
df_use.max_gooders.value_counts()

'''BASIC FREQ PLOT'''
'''LOOKING FOR ZERO OR +-1 of ZERO'''
plt.hist(df_use.max_diff_r1_days,bins=140)

'''MINERS'''
df_use['miners'] = 0
df_use.loc[df_use.min_diff_r1_days==0,'miners'] = 1
df_use.miners.value_counts()

df_use['min_gooders'] = 0
df_use.loc[df_use.min_diff_r1_days<0,'min_gooders'] = 1
df_use.min_gooders.value_counts()

'''BASIC FREQ PLOT'''
'''LOOKING FOR ZERO OR +-1 of ZERO'''
plt.hist(df_use.min_diff_r1_days,bins=140)


'''MINERS FUTHER ANALYSIS'''
df_use_gb_user_id_sum = df_use.groupby(by=df_use.user_id).sum()
df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.miners>0]['miners']













'''OLD BELOW HERE'''
'''OLD BELOW HERE'''
'''OLD BELOW HERE'''
'''GENERATE BASIC DATA STRUCTURE'''
df_S_in = pd.read_csv(f'{datadir}/AllSubs_HighestLowest_r1_r4.csv')
df_S_tmp = df_S_in[['user_id','date','date_hms','0']]
df_S_tmp.columns = ['user_id','decision_date_str','date_hms_str','prediction_date_str']
df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S_tmp['date_hms'] = df_S_tmp['date_hms_str'].map(lambda x: datetime.strptime(x.split('.')[0],'%Y-%m-%d %H:%M:%S'))
df_use = df_S_tmp.copy()

'''NEED TO  COMPUTE ROUND BC ONLY WANT ZEROs FOR ROUND X'''
df_use['round'] = 99
tmp_t = pd.to_datetime(datetime.strptime('2021-12-03T17:00:00','%Y-%m-%dT%H:%M:%S'))
df_use.loc[df_use['date_hms'] < tmp_t,'round'] = 1

tmp_t_min = pd.to_datetime(datetime.strptime('2021-12-03T17:00:00','%Y-%m-%dT%H:%M:%S'))
tmp_t_max = pd.to_datetime(datetime.strptime('2021-12-24T17:00:00','%Y-%m-%dT%H:%M:%S'))
df_use.loc[(df_use['date_hms'] > tmp_t_min) &  (df_use['date_hms'] < tmp_t_max),'round'] = 2


tmp_t_min = pd.to_datetime(datetime.strptime('2021-12-24T17:00:00','%Y-%m-%dT%H:%M:%S'))
tmp_t_max = pd.to_datetime(datetime.strptime('2022-01-14T17:00:00','%Y-%m-%dT%H:%M:%S'))
df_use.loc[(df_use['date_hms'] > tmp_t_min) &  (df_use['date_hms'] < tmp_t_max),'round'] = 3

tmp_t_min = pd.to_datetime(datetime.strptime('2022-01-14T17:00:00','%Y-%m-%dT%H:%M:%S'))
tmp_t_max = pd.to_datetime(datetime.strptime('2022-02-04T17:00:00','%Y-%m-%dT%H:%M:%S'))
df_use.loc[(df_use['date_hms'] > tmp_t_min) &  (df_use['date_hms'] < tmp_t_max),'round'] = 4

#SORT
df_use = df_use.sort_values(by='date_hms')


'''COMPUTE MIN MAX DECISION DATES FOR ALL ROUNDS AND DIFFERENCES'''
min_r1 = datetime.strptime('20211111', "%Y%m%d")
max_r1 = datetime.strptime('20220203', "%Y%m%d")
df_use['min_r1'] = min_r1
df_use['min_diff_r1'] = df_use['min_r1'] - df_use['prediction_date'] 
df_use['min_diff_r1_days'] = df_use.min_diff_r1.map(lambda x: x.days)
df_use['max_r1'] = max_r1
df_use['max_diff_r1'] = df_use['prediction_date'] - df_use['max_r1']
df_use['max_diff_r1_days'] = df_use.max_diff_r1.map(lambda x: x.days)


min_r2 = datetime.strptime('20211203', "%Y%m%d")
max_r2 = datetime.strptime('20220225', "%Y%m%d")
df_use['min_r2'] = min_r2
df_use['min_diff_r2'] = df_use['min_r2'] - df_use['prediction_date']
df_use['min_diff_r2_days'] = df_use.min_diff_r2.map(lambda x: x.days)
df_use['max_r2'] = max_r2
df_use['max_diff_r2'] = df_use['prediction_date'] - df_use['max_r2']
df_use['max_diff_r2_days'] = df_use.max_diff_r2.map(lambda x: x.days)

min_r3 = datetime.strptime('20211224', "%Y%m%d")
max_r3 = datetime.strptime('20220318', "%Y%m%d")
df_use['min_r3'] = min_r3
df_use['min_diff_r3'] = df_use['min_r3'] - df_use['prediction_date']
df_use['min_diff_r3_days'] = df_use.min_diff_r3.map(lambda x: x.days)
df_use['max_r3'] = max_r3
df_use['max_diff_r3'] = df_use['prediction_date'] - df_use['max_r3']
df_use['max_diff_r3_days'] = df_use.max_diff_r3.map(lambda x: x.days)

min_r4 = datetime.strptime('20220114', "%Y%m%d")
max_r4 = datetime.strptime('20220408', "%Y%m%d")
df_use['min_r4'] = min_r4
df_use['min_diff_r4'] = df_use['min_r4'] - df_use['prediction_date']
df_use['min_diff_r4_days'] = df_use.min_diff_r4.map(lambda x: x.days)
df_use['max_r4'] = max_r4
df_use['max_diff_r4'] = df_use['prediction_date'] - df_use['max_r4']
df_use['max_diff_r4_days'] = df_use.max_diff_r4.map(lambda x: x.days)


'''NEED SIMPLE METHOD TO DETECT MAX-MINers'''

'''MAXERS'''
df_use['maxers'] = 0
df_use.loc[(df_use.max_diff_r1_days==0) & (df_use['round']==1),'maxers'] = 1
df_use.loc[(df_use.max_diff_r2_days==0) & (df_use['round']==2),'maxers'] = 1
df_use.loc[(df_use.max_diff_r3_days==0) & (df_use['round']==3),'maxers'] = 1
df_use.loc[(df_use.max_diff_r4_days==0) & (df_use['round']==4),'maxers'] = 1
df_use.maxers.value_counts()

df_use['max_gooders'] = 0
df_use.loc[(df_use.max_diff_r1_days<0) & (df_use['round']==1),'max_gooders'] = 1
df_use.loc[(df_use.max_diff_r2_days<0) & (df_use['round']==2),'max_gooders'] = 1
df_use.loc[(df_use.max_diff_r3_days<0) & (df_use['round']==3),'max_gooders'] = 1
df_use.loc[(df_use.max_diff_r4_days<0) & (df_use['round']==4),'max_gooders'] = 1
df_use.max_gooders.value_counts()

'''BASIC FREQ PLOT'''
'''LOOKING FOR ZERO OR +-1 of ZERO'''
plt.hist(df_use.max_diff_r1_days[(df_use['round']==1)],bins=140)
plt.hist(df_use.max_diff_r2_days[(df_use['round']==2)],bins=140)
plt.hist(df_use.max_diff_r3_days[(df_use['round']==3)],bins=140)
plt.hist(df_use.max_diff_r4_days[(df_use['round']==4)],bins=140)

'''MINERS'''
df_use['miners'] = 0
df_use.loc[(df_use.min_diff_r1_days==0) & (df_use['round']==1),'miners'] = 1
df_use.loc[(df_use.min_diff_r2_days==0) & (df_use['round']==2),'miners'] = 1
df_use.loc[(df_use.min_diff_r3_days==0) & (df_use['round']==3),'miners'] = 1
df_use.loc[(df_use.min_diff_r4_days==0) & (df_use['round']==4),'miners'] = 1
df_use.miners.value_counts()

df_use['min_gooders'] = 0
df_use.loc[(df_use.min_diff_r1_days<0) & (df_use['round']==1),'min_gooders'] = 1
df_use.loc[(df_use.min_diff_r2_days<0) & (df_use['round']==2),'min_gooders'] = 1
df_use.loc[(df_use.min_diff_r3_days<0) & (df_use['round']==3),'min_gooders'] = 1
df_use.loc[(df_use.min_diff_r4_days<0) & (df_use['round']==4),'min_gooders'] = 1
df_use.min_gooders.value_counts()

'''BASIC FREQ PLOT'''
'''LOOKING FOR ZERO OR +-1 of ZERO'''
plt.hist(df_use.min_diff_r1_days[(df_use['round']==1)],bins=140)
plt.hist(df_use.min_diff_r2_days[(df_use['round']==2)],bins=140)
plt.hist(df_use.min_diff_r3_days[(df_use['round']==3)],bins=140)
plt.hist(df_use.min_diff_r4_days[(df_use['round']==4)],bins=140)

'''RESULTS: 
1.  NO MAXERS
2.  72 min judgements over full data over 4 rounds for the four low point question.
'''

'''MINERS FUTHER ANALYSIS'''
df_use_gb_user_id_sum = df_use.groupby(by=[df_use.user_id,df_use['round']]).sum()
df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.miners>0]['miners']
'''FROM THIS TABLE WE HAVE THE ANSWER:
1. All in round two, except for one S (118777) with one judgment
2. Of the Active Ss:
112076 made 21 miners in round 2
112197 made 12 miners in round 2
'''


#EOF