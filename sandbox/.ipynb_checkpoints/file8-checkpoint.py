import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

S_1 = pd.read_csv('plot_this_109742_highest.csv',index_col=0,header=0,names=['109742'])
S_1.index = pd.DatetimeIndex(S_1.index)

S_2 = pd.read_csv('plot_this_112076_highest.csv',index_col=0,header=0,names=['112076'])
S_2.index = pd.DatetimeIndex(S_2.index)

S_3 = pd.read_csv('plot_this_112197_highest.csv',index_col=0,header=0,names=['112197'])
S_3.index = pd.DatetimeIndex(S_3.index)

S_4 = pd.read_csv('plot_this_114156_highest.csv',index_col=0,header=0,names=['114156'])
S_4.index = pd.DatetimeIndex(S_4.index)

S_5 = pd.read_csv('plot_this_115725_highest.csv',index_col=0,header=0,names=['115725'])
S_5.index = pd.DatetimeIndex(S_5.index)

S_6 = pd.read_csv('plot_this_LowSubs_highest.csv',index_col=0,header=0,names=['Lows'])
S_6.index = pd.DatetimeIndex(S_6.index)

#THIS DATAFRAME IS FUCKED UP, NOT QUITE WORK AS EXPECTED
S_tmp = pd.merge(S_1,S_2,left_index=True,right_index=True,how='outer')
S_tmp = pd.merge(S_tmp,S_3,left_index=True,right_index=True,how='outer')
S_tmp = pd.merge(S_tmp,S_4,left_index=True,right_index=True,how='outer')
S_tmp = pd.merge(S_tmp,S_5,left_index=True,right_index=True,how='outer')
S_tmp = pd.merge(S_tmp,S_6,left_index=True,right_index=True,how='outer')
df = S_tmp.copy()

'''HIGH LEVEL PLOT GENERAL PATTERN'''
plt.plot(S_1,label='109742')
plt.plot(S_2,label='112076')
plt.plot(S_3,label='112197')
plt.plot(S_4,label='114156')
plt.plot(S_5,label='115725')
#plt.plot(S_6,label='Others')
#plt.xlabel('Date')
#lt.ylabel('Best Prior (Median of Dist.)')
#lt.xticks(rotation=30)
#lt.legend()
#plt.savefig('High_level_Best_Prior_over_Decision_Time.png')

plt.scatter(S_1.index,S_1)
plt.scatter(S_2.index,S_2)
plt.scatter(S_3.index,S_3)
plt.scatter(S_4.index,S_4)
plt.scatter(S_5.index,S_5)
#plt.scatter(S_6.index,S_6)
plt.xlabel('Date')
plt.ylabel('Best Prior (Median of Dist.)')
plt.xticks(rotation=30)
plt.legend()
plt.axvline(x=datetime.strptime('2021-12-03','%Y-%m-%d'),c='b',dashes=(2,2,2,2),linewidth=0.7)
plt.axvline(x=datetime.strptime('2021-12-24','%Y-%m-%d'),c='b',dashes=(2,2,2,2),linewidth=0.7)
plt.axvline(x=datetime.strptime('2022-01-14','%Y-%m-%d'),c='b',dashes=(2,2,2,2),linewidth=0.7)
#plt.savefig('High_level_Best_Prior_over_Decision_Time_highest.png',dpi=250)
plt.savefig('High_level_Best_Prior_over_Decision_Time_highest_only_freqSs.png',dpi=250)


#EOF