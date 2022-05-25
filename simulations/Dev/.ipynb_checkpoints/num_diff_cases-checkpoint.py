import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle
from scipy import stats
import matplotlib.font_manager as fm   
from pylab import cm
import matplotlib as mpl

from imp import reload
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/bayes_pack_testing')
import bayes
import b_fit

reload(bayes)

cases_data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/InputData'

'''EPI DATA'''
vdh = pd.read_csv(f'{cases_data_in}/VDH-COVID-19-PublicUseDataset-Cases.csv',parse_dates=['report_date'])

vdh = vdh.groupby(['report_date'])['total_cases'].sum().diff().rolling(window=7).mean().round()
vdh_use = vdh['11-30-2021':'01-14-2022'].copy()
vdh_use.name = 'use'

x_poly = np.arange(len(vdh_use))+1
vdh_poly = np.polyfit(x_poly, vdh_use, deg=15)
plt.plot(vdh_poly)
vdh_poly_values = np.polyval(vdh_poly, x_poly)
plt.plot(vdh_poly_values)

v_x1 = np.diff(vdh_poly_values)
v_x1_S = pd.Series(v_x1,index=vdh_use.index[1:])

v_x2 = np.diff(v_x1)
v_x2 = np.insert(v_x2,0,0,axis=0)
plt.plot(v_x1); plt.plot(v_x2);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

v_x2_S = pd.Series(v_x2,index=vdh_use.index[2:])
plt.plot(v_x2_S); plt.plot(vdh_use[2:]); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

plt.plot(v_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)


'''ADD THE HUMAN DATA'''
'''ADD THE HUMAN DATA'''
data_in = '/Users/biocomplexity/Projects/SocioCognitiveModeling/Metaculus_CogModeling/simulations/Preprocessing'
infile_name = 'sim_out_highest'

df_s_freqs = pd.read_csv(f'{data_in}/Final_Cleaned_High_R1_R3_S_freqs.csv')
s_freq_threshold = 15
s_S = df_s_freqs.loc[df_s_freqs.decision_date>s_freq_threshold,'user_id']
s_list = s_S.to_list()
group_list = ['all_s'] + s_list


'''PUT ALL DATA FOR EACH GROUP IN LIST'''
catch_groups = []
for i in group_list:
    '''ANALYZE All Ss'''
    infile_group = i

    '''GRAB DATA'''
    S_all = pd.read_pickle(f'S_all_{infile_name}_{infile_group}.csv')
    
    catch_groups.append(S_all)


gp_no = group_list[0]
i = catch_groups[0]
'''aVE OVER DAY'''
grouped = i.groupby(level=0)


'''
HUMAN DATA 
FOR DIFFERENTIIATION
'''

#TRY NON SMOOTHED DATA
hum_use = grouped.hum.mean()

x_poly = np.arange(len(hum_use))+1
hum_poly = np.polyfit(x_poly, hum_use, deg=15)#deg 15 worked
plt.plot(hum_poly)
hum_poly_values = np.polyval(hum_poly, x_poly)
plt.plot(hum_poly_values)
plt.plot(hum_use)

h_x1 = np.diff(hum_poly_values)
h_x1_S = pd.Series(h_x1,index=hum_use.index[1:])

h_x2 = np.diff(h_x1)
h_x2 = np.insert(h_x2,0,0,axis=0)
plt.plot(h_x1); plt.plot(h_x2);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

h_x2_S = pd.Series(h_x2,index=hum_use.index[2:])
plt.plot(h_x2_S); plt.plot(hum_use[2:]); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

plt.plot(h_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.savefig('h_x2_S.png', dpi=300, transparent=False, bbox_inches='tight')

plt.plot(v_x2_S);  plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.savefig('v_x2_S.png', dpi=300, transparent=False, bbox_inches='tight')




'''PLOT HUMAN AND EPI TOGETHER'''
norm_const = 0.2

v_const = 0.01; h_const = 5; plt.plot(v_x1*v_const); plt.plot(h_x1*h_const); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

v_const = 0.05; h_const = 5; plt.plot(v_x2*v_const); plt.plot(h_x2*h_const); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)

#EPI ACC, HUMAN VEL
v_const = 0.05; h_const = 5; plt.plot(v_x2*v_const,label='Epi Acc.'); plt.plot(h_x1*h_const, label='Hum Vel.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.legend()
plt.savefig('Epi_Acc_w_Hum_Vel.png',dpi=200)

#EPI VEL, HUMAN ACC
v_const = 0.05; h_const = 5; plt.plot(v_x1*v_const,label='Epi Vel.'); plt.plot(h_x2*h_const, label='Hum Acc.'); plt.axhline(y=0, c='r',dashes=(2,2,2,2),linewidth=2)
plt.legend()
plt.savefig('Epi_Vel_w_Hum_Acc.png',dpi=200)

#SCATTERS
df_b = pd.DataFrame([v_x1_S,h_x2_S]).T
plt.scatter(df_b[0],df_b[1])

df_b_2 = pd.DataFrame([v_x2_S,h_x1_S]).T
plt.scatter(df_b_2[0]*v_const,df_b_2[1]*h_const)

#USE df_b for phase space.
df_b.columns = ['epi_vel','hum_acc']

#MAKE X and Y
S_x = v_x1.copy()
S_x = np.insert(S_x,0,0,axis=0) #PAD FIRST
S_x = S_x[:-1].copy() #REMOVE LAST
S_x = pd.Series(S_x,index=df_b.index)

S_y = h_x2.copy()
S_y = np.insert(S_y,0,0,axis=0) #PAD FIRST
S_y = S_y[:-1].copy() #REMOVE LAST
S_y = pd.Series(S_y,index=df_b.index)

df_b['S_x'] = S_x
df_b['S_y'] = S_y

thing = v_x1.copy()
catch_arr = np.array([])
for i in range(1,len(thing)):
    catch_arr = np.append(catch_arr,thing[i]-thing[i-1])
S_dx = catch_arr.copy()
S_dx = np.insert(S_dx,0,0,axis=0)
df_b['S_dx'] = S_dx

thing = h_x2.copy()
catch_arr = np.array([])
for i in range(1,len(thing)):
    catch_arr = np.append(catch_arr,thing[i]-thing[i-1])
S_dy = catch_arr.copy()
S_dy = np.insert(S_dy,0,0,axis=0)
df_b['S_dy'] = S_dy



plt.arrow(df_b.S_x[2], df_b.S_x[2], df_b.S_dx[2], df_b.S_dy[2],head_width = 1, width = 0.001)

for i in range(2,len(df_b)):
    plt.arrow(df_b.S_x[i], df_b.S_y[i], df_b.S_dx[i], df_b.S_dy[i],head_width = .4, width = 0.1)

for i in range(1,15):
    plt.arrow(df_b.S_x[i]*0.05, df_b.S_y[i]*5, df_b.S_dx[i]*0.05, df_b.S_dy[i]*5,head_width = 1, width = 0.1)

for i in range(15,30):
    plt.arrow(df_b.S_x[i]*0.05, df_b.S_y[i]*5, df_b.S_dx[i]*0.05, df_b.S_dy[i]*5,head_width = .5, width = 0.1)

for i in range(30,45):
    plt.arrow(df_b.S_x[i]*0.05, df_b.S_y[i]*5, df_b.S_dx[i]*0.05, df_b.S_dy[i]*5,head_width = .5, width = 0.1)

for i in range(40,45):
    plt.arrow(df_b.S_x[i]*0.05, df_b.S_y[i]*5, df_b.S_dx[i]*0.05, df_b.S_dy[i]*5,head_width = .5, width = 0.1)


plt.plot(np.arange(10))
for i in range(2,15):
    plt.plot(np.arange(10))
    plt.xlim([-200,200])
    plt.ylim([-2,2])
    plt.annotate("", xy=(df_b.S_x[i],df_b.S_y[i]), xytext=(float(df_b.S_dx[i]),float(df_b.S_dy[i])), arrowprops=dict(arrowstyle="->")  )

    
plt.annotate("", xy=(0.5, 0.5), xytext=(0, 0),arrowprops=dict(arrowstyle="->"))
plt.annotate("", xy=(.7, 0.7), xytext=(0.5, 0.5),arrowprops=dict(arrowstyle="->"))

plt.annotate("", xy=(S_x[2],S_y[2]), xytext=(0,0), arrowprops=dict(arrowstyle="->"))

#NO ARROWS
x = df_b.S_x
y = df_b.S_y

fig, ax = plt.subplots()
ax.scatter(x,y)
ax.plot(x,y)


'''THIS IS THE GOOD ONE, BUT NOT QUITE PUB QUALITY'''
x_min = min(df_b.S_x)-50
x_max = max(df_b.S_x)+50
y_min = min(df_b.S_y)+4#SPECIAL
y_max = max(df_b.S_y)+1
             

#ARROWS
x = df_b.S_x.iloc[2:10].values
y = df_b.S_y.iloc[2:10].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max]); plt.show()

#ARROWS
x = df_b.S_x.iloc[9:20].values
y = df_b.S_y.iloc[9:20].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max]); plt.show()

#ARROWS
x = df_b.S_x.iloc[19:32].values
y = df_b.S_y.iloc[19:32].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max]); plt.show()

#ARROWS
x = df_b.S_x.iloc[31:45].values
y = df_b.S_y.iloc[31:45].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max]);plt.show()


'''ALL TOGETHER'''
#ARROWS
x = df_b.S_x.iloc[2:].values
y = df_b.S_y.iloc[2:].values

u = np.diff(x)
v = np.diff(y)
pos_x = x[:-1] + u/2
pos_y = y[:-1] + v/2
norm = np.sqrt(u**2+v**2) 

fig, ax = plt.subplots(); ax.plot(x,y, marker="o"); ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid"); ax.set_xlim([x_min,x_max]); ax.set_ylim([y_min,y_max]); plt.show()





'''DEV'''
'''
ISSUE: JUSTIFYINIG POLYNOMIAL OF HIGHER DEG
0. deg of poly does not change epi transformation
1. Test if lower deg poly on human will still give same relation to epii
2. If yes, use it.
3. If no, then can justify using higher bc it fits epi, but then 
must show not spurious--> test random field (using var of hum) and fit same set 
of higher degree poly and match to epi.  If not get same fit, then we are ok. 
Will use Monte Carlo with 1000 runs to give probability that we can randomly get
a harmonic oscilllator.
X. May use family  of poly for match to epi
X. May use more advanced regression methods with penalization or regularization.
X. May check sensitivity of poly to values
CONVINCING WAY TO ADJUDICATE THE USE OF HIGH DEG POLYNOMIALL


'''

#deg_list = [10,11,12,13,14,15,16,17]
#deg_list = [10,11,12,13]
deg_list = [14,15,16]
for i in deg_list:
    x_poly = np.arange(len(hum_use))+1
    hum_poly = np.polyfit(x_poly, hum_use, deg=i,full=True)#deg 15 worked
    hum_poly_values = np.polyval(hum_poly[0], x_poly)
    plt.plot(hum_poly_values)
    print('RESIDUALS SUM: ',hum_poly[1])
    
plt.scatter(np.arange(len(hum_use)),np.array(hum_use))
plt.plot(np.array(hum_use))
plt.plot(np.array(grouped.hum.mean().rolling(4).mean()),color='red',label='prior',dashes=(0,2,2,2))


for i in deg_list:
    x_poly = np.arange(len(vdh_use))+1
    vdh_poly = np.polyfit(x_poly, vdh_use, deg=15)
    vdh_poly_values = np.polyval(vdh_poly, x_poly)
    plt.plot(vdh_poly_values)
#EOF

hum_poly = np.polyfit(x_poly, hum_use, deg=15,full=True)
hum_poly[1]

plt.plot(hum_use)
plt.scatter(hum_use.index[1:],hum_use[1:])
