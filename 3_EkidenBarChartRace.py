import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
import numpy as np
from tqdm import tqdm
import gc
import random
import glob

if not os.path.exists("image"):
    os.mkdir("image")

if not os.path.exists("video"):
    os.mkdir("video")
    
data_file = 'data/4_preprocessed_final.csv'
title_fontsize = 10
footnote2 = 'Data Source: https://www.hakone-ekiden.jp/record/'
footnote1 = 'Source Code: https://github.com/Simple-Charts/EkidenBarChartRace'
footnote_fontsize = 4
xlabel_text = 'station'
xlabel_fontsize = 5
tick_fontsize = 5
tick_offset_fontsize = 6
fontname = "Meiryo"
time_fontsize = 16
frames = 5
image_dpi = 300
video_height = 500
video_width = 800
video_fps = 15
time_steps = 299

def sec_to_hms(s):
    h = s // (60*60)
    s %= (60*60)
    m = s // 60
    s %= 60
    return "%02i:%02i:%02i" % (h, m, s)

df_full = pd.read_csv(data_file)
total_row = df_full.loc[:,'second'].index[df_full.loc[:,'second']=='total'].tolist()
years = [df_full.loc[k,'year'] for k in total_row]
total_row = [-1] + total_row
labels = df_full.columns[2:]
colors = [cm.gist_rainbow(i / len(labels)) for i in range(len(labels))]
colors = [(color[0], color[1], color[2], 0.75) for color in colors]
random.seed(0)
random.shuffle(colors)
mov_no = 0
for v in range(len(years)):
    df = df_full.loc[total_row[v]+1:total_row[v+1]-1,:].drop('year', axis=1)
    df = df.reset_index(drop=True)
    df['second'] = df['second'].astype('int')
    df_total = df_full.loc[total_row[v]+1:total_row[v+1],:].drop('year', axis=1)
    title = 'Hakone Ekiden Marathon Relay / ' + str(years[v])
    df_total = df_total.set_index('second')
    df_total = df_total.iloc[-1,:][df_total.iloc[-1,:]>0].astype('int')
    df_final_rank = df_total.rank(method='first')
    df_final_rank = df.shape[1] - df_final_rank.astype('int')
    df = df.fillna(0)
    df.index = df.index * frames
    df_value = df.reindex(range((len(df)-1)*frames+1))    
    df_rank = df_value.copy()
    df_rank['second'] = df_rank['second'].fillna(method='ffill')
    df_rank = df_rank.set_index('second')
    df_rank = df_rank.rank(axis=1, method='first')
    df_value = df_value.interpolate()
    df_value = df_value.set_index('second')
    finished_team = []
    finished_rank = []
    finished_time = []
    for i in range(len(df_value)):
        zero_value = df_value.iloc[i].index[df_value.iloc[i] == 0.0]
        finished = list(df_value.iloc[i].index[df_value.iloc[i] == 10.])
        df_rank.iloc[i][zero_value] = 0
    finished_team = list(df_final_rank.index)
    finished_time = list(df_total)
    finished_rank = list(df_final_rank)
    i=0
    for b in finished_team:
        j=0
        for c in df_value.loc[:,b]:
            if c == 10.0:
                df_rank.loc[:,b].iloc[j] = finished_rank[i]
            j=j+1
        i=i+1
    df_rank = df_rank.interpolate()
    ranks = len(df_rank.sum()[df_rank.sum()>0])
    for i in tqdm(range(len(df_value))):
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=144)
        plt.rcParams["font.family"] = fontname 
        ax.barh(y = df_rank.iloc[i], width = df_value.iloc[i], color = colors, tick_label = labels)
        ax.set_xlabel(xlabel_text, fontsize = xlabel_fontsize)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labelsize = tick_fontsize)
        ax.tick_params(axis='y', labelsize = tick_fontsize)
        ax.grid(which='major', axis='x', linestyle='-')
        ax.set_axisbelow(True)
        [spine.set_visible(False) for spine in ax.spines.values()]
        ax.set_ylim(len(labels) - (ranks - 0.5), len(labels) + 0.5)
        ax.set_xlim(0,)
        x_start, x_end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(x_start, x_end, 1))    
        plt.subplots_adjust(left=0.35, right=0.9, bottom=0.1, top=0.75)    
        plt.suptitle(title, y=0.95, fontsize = title_fontsize)
        ax.text(1, -0.12, sec_to_hms(int(df_value.index[i])), transform = ax.transAxes, size = time_fontsize, ha='right', weight=750)
        max_value = df_value.iloc[i].max()
        q = 0
        for z in ax.get_yticklabels():
            if ax.patches[q].get_width() == 10:
                plt.text(max_value, ax.patches[q].get_y() + ax.patches[q].get_height()/2, sec_to_hms(finished_time[finished_team.index(z.get_text())]), va='center', fontsize = tick_fontsize, fontweight ='bold', color ='blue')
            q = q + 1
        ax.text(-0.6, -0.1, footnote1, transform = ax.transAxes, size = footnote_fontsize, ha='left')
        ax.text(-0.6, -0.14, footnote2, transform = ax.transAxes, size = footnote_fontsize, ha='left')
        plt.savefig("image/img"+str(i).zfill(4)+".png", dpi=image_dpi)
        plt.clf()
        plt.close()

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter('video/video'+ str(mov_no).zfill(4) +'.mp4', fourcc, fps = video_fps, frameSize = (video_width, video_height))
    for i in tqdm(range(len(df_value))):
        img = cv2.imread('image/img%04d.png' % i)
        img = cv2.resize(img, (video_width, video_height))
        video.write(img)
    video.release()
    mov_no = mov_no + 1
    del video
    gc.collect()

def comb_movie(movie_files,out_path):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    movie = cv2.VideoCapture(movie_files[0])
    fps = movie.get(cv2.CAP_PROP_FPS)
    height = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = movie.get(cv2.CAP_PROP_FRAME_WIDTH)
    out = cv2.VideoWriter(out_path, int(fourcc), fps*2, (int(width), int(height)))
    for movies in (movie_files):
        print(movies)
        movie = cv2.VideoCapture(movies)
        if movie.isOpened() == True: 
            ret, frame = movie.read() 
        else:
            ret = False
        while ret:
            out.write(frame)
            ret, frame = movie.read()

files = sorted(glob.glob('video/*.mp4'))
out_path = "result.mp4"
comb_movie(files,out_path)