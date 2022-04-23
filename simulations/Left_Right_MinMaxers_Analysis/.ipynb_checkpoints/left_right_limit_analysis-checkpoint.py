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
outdir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Left_Right_MinMaxers_Analysis'
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

q_list = [9018, 9426, 8570, 8799, 9016]

catch_miners_maxers_analy = []
for i in q_list:
    df_S_in = pd.read_csv(f'{datadir}/AllSubs_q{i}.csv')
    df_S_tmp = df_S_in[['user_id','date','0']].copy()
    df_S_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
    df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
    df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
    df_use = df_S_tmp.copy()
    df_use = df_use.sort_values(by='decision_date')
    
    if i == 9018: 
        min_rX = min_r3 
        max_rX = max_r3
    if i == 9426:
        min_rX = min_r4 
        max_rX = max_r4
    if i == 8570:
        min_rX = min_r1 
        max_rX = max_r1
    if i == 8799:
        min_rX = min_r2 
        max_rX = max_r2
    if i == 9016:
        min_rX = min_r3 
        max_rX = max_r3
        
    df_use['min_x'] = min_rX
    df_use['min_diff'] = df_use['min_x'] - df_use['prediction_date'] 
    df_use['min_diff_days'] = df_use.min_diff.map(lambda x: x.days)
    df_use['max_x'] = max_rX
    df_use['max_diff'] = df_use['prediction_date'] - df_use['max_x']
    df_use['max_diff_days'] = df_use.max_diff.map(lambda x: x.days)

    '''MAXERS'''
    df_use['maxers'] = 0
    df_use.loc[df_use.max_diff_days==0,'maxers'] = 1
    df_use.maxers.value_counts()

    df_use['max_gooders'] = 0  
    df_use.loc[df_use.max_diff_days<0,'max_gooders'] = 1
    df_use.max_gooders.value_counts()

    '''BASIC FREQ PLOT'''
    '''LOOKING FOR ZERO OR +-1 of ZERO'''
    plt.hist(df_use.max_diff_days,bins=140)
    print(df_use.max_diff_days.min())
    print(df_use.max_diff_days.max())

    '''MINERS'''
    df_use['miners'] = 0
    df_use.loc[df_use.min_diff_days==0,'miners'] = 1
    df_use.miners.value_counts()

    df_use['min_gooders'] = 0
    df_use.loc[df_use.min_diff_days<0,'min_gooders'] = 1
    df_use.min_gooders.value_counts()

    '''BASIC FREQ PLOT'''
    '''LOOKING FOR ZERO OR +-1 of ZERO'''
    plt.hist(df_use.min_diff_days,bins=140)
    print(df_use.min_diff_days.min())
    print(df_use.min_diff_days.max())


    '''IDENTIFICATION OF Ss THAT ARE MINERS OR MAXERS'''
    df_use_gb_user_id_sum = df_use.groupby(by=df_use.user_id).sum()
    print(df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.miners>0,'miners'])
    print(df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.maxers>0,'maxers'])
    
    catch_miners_maxers_analy.append([i,
                                      ('right-min',df_use.max_diff_days.min()),
                                      ('right-max',df_use.max_diff_days.max()),
                                      ('left-min',df_use.min_diff_days.min()),
                                      ('left-max',df_use.min_diff_days.max()),
                                      'MINERS Ss',
                                      df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.miners>0,'miners'],
                                      'MAXERS Ss',
                                      df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.maxers>0,'maxers']])
#END LOOP

'''
FOR ANALYSIS:
1. loop will plot all qs together as freq
2. catch from loop has all info need.
'''
#EASY FORMAT
for i in catch_miners_maxers_analy:
    print('NEXT Q',i[0])
    print(i)

out_data = pd.DataFrame(catch_miners_maxers_analy)  
out_data[6][4]
out_data
'''OUTDATA IS THE DATA USED FOR THE MS'''







'''EXAMPLE OF ONE QUESTION AT A TIME'''
'''WONT USE THIS BUT HERE FOR REFERENCE'''
'''QUESTION 9018 RD 3 Lowest'''
df_S_in = pd.read_csv(f'{datadir}/AllSubs_q8570.csv')
df_S_tmp = df_S_in[['user_id','date','0']].copy()
df_S_tmp.columns = ['user_id','decision_date_str','prediction_date_str']
df_S_tmp['decision_date'] = df_S_tmp['decision_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_S_tmp['prediction_date'] = df_S_tmp['prediction_date_str'].map(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
df_use = df_S_tmp.copy()
df_use = df_use.sort_values(by='decision_date')

df_use['min_r3'] = min_r3
df_use['min_diff_r3'] = df_use['min_r3'] - df_use['prediction_date'] 
df_use['min_diff_r3_days'] = df_use.min_diff_r3.map(lambda x: x.days)
df_use['max_r3'] = max_r3
df_use['max_diff_r3'] = df_use['prediction_date'] - df_use['max_r3']
df_use['max_diff_r3_days'] = df_use.max_diff_r3.map(lambda x: x.days)

'''MAXERS'''
df_use['maxers'] = 0
df_use.loc[df_use.max_diff_r3_days==0,'maxers'] = 1
df_use.maxers.value_counts()

df_use['max_gooders'] = 0
df_use.loc[df_use.max_diff_r3_days<0,'max_gooders'] = 1
df_use.max_gooders.value_counts()

'''BASIC FREQ PLOT'''
'''LOOKING FOR ZERO OR +-1 of ZERO'''
plt.hist(df_use.max_diff_r3_days,bins=140)
print(df_use.max_diff_r3_days.min())
print(df_use.max_diff_r3_days.max())

'''MINERS'''
df_use['miners'] = 0
df_use.loc[df_use.min_diff_r3_days==0,'miners'] = 1
df_use.miners.value_counts()

df_use['min_gooders'] = 0
df_use.loc[df_use.min_diff_r3_days<0,'min_gooders'] = 1
df_use.min_gooders.value_counts()

'''BASIC FREQ PLOT'''
'''LOOKING FOR ZERO OR +-1 of ZERO'''
plt.hist(df_use.min_diff_r3_days,bins=140)
print(df_use.min_diff_r3_days.min())
print(df_use.min_diff_r3_days.max())

'''IDENTIFICATION OF Ss THAT ARE MINERS OR MAXERS'''
df_use_gb_user_id_sum = df_use.groupby(by=df_use.user_id).sum()
print(df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.miners>0]['miners'])
print(df_use_gb_user_id_sum.loc[df_use_gb_user_id_sum.maxers>0]['maxers'])














#EOF