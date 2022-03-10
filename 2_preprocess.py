import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os

if not os.path.exists("data"):
    os.mkdir("data")

data_file = 'data/1_raw_data.csv'
team_file = 'team.csv'
year_file = 'year.csv'
time_steps = 299

def hms_to_sec(x):
    try:
        hms = re.findall(r"\d+", x)
        sec = int(hms[0])*60*60 + int(hms[1])*60 + int(hms[2])
        return sec 
    except:
        pass
def sec_to_hms(s):
    h = s // (60*60)
    s %= (60*60)
    m = s // 60
    s %= 60
    return "%02i:%02i:%02i" % (h, m, s)

df = pd.read_csv(data_file, header=None, usecols=[0,2,4,6,8,9,12])
df_team = pd.read_csv(team_file, header=0, usecols=[1,2])
dict_team = df_team.set_index('Japanese').T.to_dict('list')
df_year = pd.read_csv(year_file, header=0, usecols=[1,2])
dict_year = df_year.set_index('tn').T.to_dict('list')
for i in tqdm(range(len(df))):
    df.iloc[i,1] = dict_team.get(df.iloc[i,1])[0]
    df.iloc[i,2] = hms_to_sec(df.iloc[i,2])
    df.iloc[i,3] = hms_to_sec(df.iloc[i,3])
    df.iloc[i,4] = hms_to_sec(df.iloc[i,4])
    df.iloc[i,6] = hms_to_sec(df.iloc[i,6])
df = df.assign(Accum=np.nan)
df.set_index(list(df.columns[[0,1]]))
group = df.groupby(list(df.columns[[0,1]]))
for idx in list(group.groups.keys()):
    r =  list(df[(df[df.columns[0]] == idx[0]) & (df[df.columns[1]] == idx[1])].index) 
    if list(df.iloc[r,5])==[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        if df.iloc[r,6].isnull().sum() == 0:
            df.iloc[r[0],7]  = df.iloc[r[0],6]
            for k in range(1,10):
                df.iloc[r[k],7]  = df.iloc[r[k-1],7] + df.iloc[r[k],6]
        else:
            df.iloc[r,7] = '*'
    else:
        df.iloc[r,7] = '#'
df.to_csv('data/2_preprocessed_data.csv',encoding='utf_8_sig')

group2 = df.groupby([df.columns[0]])
idx_full = []
index_full = []
for idx in tqdm(list(group2.groups.keys())):
    r =  list(df[(df[df.columns[0]] == idx)].index)
    df_selected = df.iloc[r,7]
    if len(df_selected[(df_selected == '*') | (df_selected == '#')].index) == 0:
        index_full.extend(list(df_selected.index))
        idx_full.append(idx)
df_full = df.iloc[index_full,:]
df_full.to_csv('data/3_preprocessed_filled_data.csv',encoding='utf_8_sig')

df_full.set_index([df.columns[0]])
group3 = df_full.groupby([df.columns[0]])
dict_ID_maxAccum = {}
for v, x in zip(group3.groups.values(), group3.groups.keys()):
    c = list(v)
    maxAccum = max(list(df_full.loc[c].iloc[:,7]))
    dict_ID_maxAccum[x] = maxAccum

df_full.set_index(list(df.columns[[0,1]]))
group4 = df_full.groupby(list(df.columns[[0,1]]))

group5 = df_full.groupby([df.columns[1]])
teams = list(group5.groups.keys())

result = pd.DataFrame(index=range((time_steps+1)*len(dict_ID_maxAccum)), columns=['year', 'second']+teams)
result.fillna(0, inplace=True)

for v, x in tqdm(zip(group4.groups.values(), group4.groups.keys()), total = len(group4.groups.values())):
    c = list(v)
    final_time = list(df_full.loc[c].iloc[:,2])[0]
    t = list(df_full.loc[c].iloc[:,7])
    t = [float(x) for x in t]
    total_time = dict_ID_maxAccum[list(x)[0]]
    r = np.linspace(0, total_time, time_steps)
    a   =  np.piecewise(r, [(r<t[0]),((r>=t[0])&(r<t[1])),
                                     ((r>=t[1])&(r<t[2])),
                                     ((r>=t[2])&(r<t[3])),
                                     ((r>=t[3])&(r<t[4])),
                                     ((r>=t[4])&(r<t[5])),
                                     ((r>=t[5])&(r<t[6])),
                                     ((r>=t[6])&(r<t[7])),
                                     ((r>=t[7])&(r<t[8])),
                                     ((r>=t[8])&(r<t[9])),(r>=t[9])],
                                    [lambda x: (1/t[0])*x,
                                     lambda x: (1/(t[1]-t[0]))*(x-t[0])+1, 
                                     lambda x: (1/(t[2]-t[1]))*(x-t[1])+2, 
                                     lambda x: (1/(t[3]-t[2]))*(x-t[2])+3, 
                                     lambda x: (1/(t[4]-t[3]))*(x-t[3])+4, 
                                     lambda x: (1/(t[5]-t[4]))*(x-t[4])+5, 
                                     lambda x: (1/(t[6]-t[5]))*(x-t[5])+6, 
                                     lambda x: (1/(t[7]-t[6]))*(x-t[6])+7, 
                                     lambda x: (1/(t[8]-t[7]))*(x-t[7])+8, 
                                     lambda x: (1/(t[9]-t[8]))*(x-t[8])+9,10])
    a = [np.round(x,2) for x in list(a)]
    r = [int(x) for x in list(r)]
    for y in range(len(a)):
        result.iloc[y+(time_steps+1)*idx_full.index(list(x)[0]), teams.index(list(x)[1])+2] = a[y]
        result.iloc[y+(time_steps+1)*idx_full.index(list(x)[0]), 1] = r[y]
        result.iloc[y+(time_steps+1)*idx_full.index(list(x)[0]), 0] = dict_year[list(x)[0]][0]
    result.iloc[len(a)+(time_steps+1)*idx_full.index(list(x)[0]), teams.index(list(x)[1])+2] = final_time
    result.iloc[len(a)+(time_steps+1)*idx_full.index(list(x)[0]), 1] = "total"
    result.iloc[len(a)+(time_steps+1)*idx_full.index(list(x)[0]), 0] = dict_year[list(x)[0]][0]

result.to_csv('data/4_preprocessed_final.csv',encoding='utf_8_sig',index=False)