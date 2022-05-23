import pandas as pd
import numpy as np
import math
from ast import literal_eval
import collections
import os, glob, pickle
import librosa
import pathlib
from pydub import AudioSegment
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import time

import demo_analysis_new

NUM_WORKERS=2

match_len={
    1:4,
    2:27,
    3:37,
    4:31}


emotion_with_keys={
  1:'интерес',
  2:'радость',
  3:'удивление',
  4:'горе',
  5:'гнев',
  6:'отвращение',
  7:'презрение',
  8:'страх',
  9:'стыд',
 10:'вина'
}


def get_match_info(n_match):
    if n_match == 1:
        parsed_demo = './demos/3248aa5e-b344-40f5-8f83-4988a3b7141b_de_vertigo_128.csv'
    if n_match == 2:
        parsed_demo ='./demos/83fdd578-cb07-4c86-abb1-304cb0328b78_de_overpass_128.csv'
    if n_match == 3:
        parsed_demo = './demos/3e9849b7-304d-4017-96bc-41e7f0ce6a4e_de_vertigo_128.csv'
    if n_match == 4:
        parsed_demo = './demos/b504a2f2-82b6-4385-b0f3-ef9b88949655_de_mirage_128.csv'
        
    df = pd.read_csv(parsed_demo)
    tickrate=128 #default
    rounds_list = demo_analysis_new.get_round_stat(df,tickrate)

    return rounds_list

def get_players(n_match):
  
    players=['incr0ss','Softcore','humllet','faceitkirjke','SL4VAMARL0W']
    if n_match in [1,2]:
        players+=['___Tox1c___','giena1337','TheDefenderr','HOoL1GAN_','DENJKEZOR666'] #- VTB
    elif n_match==3:
        players+=['zhenn--','riddle','savagekx','Ka1n___','_SEGA'] #- GBCB
    elif n_match==4:
        players+=['zhenn--','riddle','savagekx','Ka1n___','RubinskiyRV'] #- GBCB
    
    return players

def get_game_context():
    game_context={}
    for n_match in range(1,5):
        rounds_list = get_match_info(n_match)
        rounds={}
        players=get_players(n_match)
        for n_round,round_ in enumerate(rounds_list):
            round_data = round_[0].copy()
            round_data['users_self'] = round_data.parameters.apply(literal_eval).apply(lambda x: x.get('userid'))
            round_data['users_self'] = round_data['users_self'].apply(lambda x: players.index(x.split()[0]) if x and x.split()[0] in players else None)
            round_data['users_attacker'] = round_data.parameters.apply(literal_eval).apply(lambda x: x.get('attacker'))
            round_data['users_attacker'] = round_data['users_attacker'].apply(lambda x: players.index(x.split()[0]) if x and x.split()[0] in players else None)
            
            round_data['ms']=(((round_data['tick'] - round_data.tick.iloc[0])/128)*1000)
            rounds[n_round]=round_data
        game_context[n_match]=rounds
    return game_context


#разбиение всех аудио по 3 сек в соответствии с csv разметкой
def split_audio(path_to_audio,path_to_splitted_audio):
    for path in pathlib.Path(path_to_audio).iterdir():
        if path.is_dir():
            player_number = str(path)[str(path).rfind('/')+1:].split('_')[0]
            for path_in in pathlib.Path(path).iterdir():
                str_path = str(path_in)
                name = str_path[str_path.rfind('/')+1:]
                sound = AudioSegment.from_wav(path_in)
                i=0
                while i<=len(sound):
                    if i+3000>len(sound):
                        cut = sound[i:len(sound)+1]
                        if not os.path.exists(path_to_splitted_audio+'/'+player_number+'_'+name[:-4]+ f'_{i}_{len(sound)}.wav'):
                            cut.export(path_to_splitted_audio+'/'+player_number+'_'+name[:-4]+ f'_{i}_{len(sound)}.wav', format="wav")
                        break
                    cut = sound[i:i+3000]
                    if not os.path.exists(path_to_splitted_audio+'/'+player_number+'_'+name[:-4]+ f'_{i}_{i+3000}.wav'):
                        cut.export(path_to_splitted_audio+'/'+player_number+'_'+name[:-4]+ f'_{i}_{i+3000}.wav', format="wav")
                    i+=3000

def get_dict_with_emotions(file_path): #keys: match -> n_round -> num_player value: dataframe
    all_emt_dict={}
    col_emt_names=["start", "end", "emt_est_1", "str_emt_est_1", "emt_est_2", "str_emt_est_2", "emt_est_3", "str_emt_est_3"]
    for n_match in range(4):
        match_emt_cl={}
        for n_round in range(1,match_len[n_match+1]+1):

            players_emt={}
            res_emt={}
            for num_player in range(10):
                if glob.glob(file_path+f'/{num_player}_match{n_match+1}_round{n_round}.csv'):
                    str_path=glob.glob(file_path+f'/{num_player}_match{n_match+1}_round{n_round}.csv')[0]
                    df_s = pd.read_csv(str_path,names=col_emt_names)
                    res_emt[num_player]=df_s

            match_emt_cl[n_round]=res_emt
        all_emt_dict[n_match]=match_emt_cl 
    return all_emt_dict

def find_majority(est_with_str):
    votes=[i[0] for i in est_with_str]
    strength=[i[1] for i in est_with_str]
    vote_count = Counter(votes)
    most_ = vote_count.most_common(1)
    if (most_[0][1]>=2)and(most_[0][0]>0):
        ids = [ind for ind in range(len(votes)) if votes[ind] == most_[0][0]]
        mean_str = round(np.array([strength[i] for i in ids]).mean(),1)
        return [most_[0][0],mean_str]
    else:
        return [-1,-1]

def get_emotions(df): 
    
    player_emt = df.copy()
    for i in range(1,4):
        player_emt[f'est_{i}'] = list(zip(player_emt[f'emt_est_{i}'], player_emt[f'str_emt_est_{i}']))

    ss=[]
    for i in player_emt[['est_1','est_2','est_3']].values:
        ss.append(find_majority(i))

    player_emt['emt']=np.asarray(ss)[:,0]
    player_emt['emt'] = player_emt['emt'].apply(int)
    player_emt['str_emt']=np.asarray(ss)[:,1]
    
    emt_in_time={}
    for i in player_emt[['start','emt']].query('emt>0').emt.unique():
        emt_in_time[i]=[j[0] for j in player_emt[['start','emt']].query('emt>0').values if j[1]==i]
    return emt_in_time

def convert_dict(all_emt_dict):
    full_emt={} #key - emotion_type: value - [n_player,n_match,n_round,start_time]
    for key in emotion_with_keys.keys():
        full_emt[key]=[]
    
    for n_match in range(4):
        for n_round in range(match_len[n_match+1]+1):
            for n_player in range(10):
                if all_emt_dict.get(n_match) is not None and \
                   all_emt_dict.get(n_match).get(n_round) is not None and \
                   all_emt_dict.get(n_match).get(n_round).get(n_player) is not None:
                        emt_in_time = get_emotions(all_emt_dict.get(n_match).get(n_round).get(n_player))
                        for e, start_time in emt_in_time.items():
                            full_emt[e].append([n_player,n_match,n_round,start_time])
    return full_emt

def split_annotations(full_emt,path_to_audio,test_size=0.2):
    train_annotations_list = []
    val_annotations_list = []
    
    X = []
    y = []
    info=[]
    
    for key_emt, values in full_emt.items():
        for vals in values:
            n_player,n_match,n_round,list_start_time=vals
            for start_time in list_start_time:
                file = glob.glob(path_to_audio+f'/{n_player}_match{n_match+1}_round{n_round}_{start_time}*.wav')
                X.append(file[0])
                y.append(key_emt)
                info.append([n_player,n_match+1,n_round,start_time])
                
    #не учитываем "стыд"       
    ind_to_del = y.index(9)
    y.pop(ind_to_del)
    X.pop(ind_to_del)
    info.pop(ind_to_del)
    
    X_train, X_val, y_train, y_val, info_train, info_val = train_test_split(X, y, info, test_size=test_size, random_state=42, shuffle=True, stratify=y)
    train_annotations_list = list(zip(X_train, y_train,info_train))
    val_annotations_list = list(zip(X_val, y_val, info_val))
    
    return train_annotations_list, val_annotations_list



def display_emt(full_dict_with_emt):
    for key, value in full_dict_with_emt.items():
      print(key,'-',np.array([len(i[3]) for i in value]).sum(),'   ',emotion_with_keys[key])


def create_onehot_tensor(label):
    y_onehot = torch.zeros(len(emotion_with_keys))
    y_onehot[label-1]=1
    return y_onehot


def get_context_vector(info,game_context):
    context_vector = np.zeros(12)
    n_player,n_match,n_round,start_time = info
    bomb_interaction_events=['bomb_pickup','bomb_beginplant','bomb_planted','bomb_exploded','bomb_begindefuse','bomb_defused','bomb_abortplant']
    if n_round>=2:
        end_time = game_context[n_match][n_round-2].iloc[-1].ms
        if start_time==0 and n_round-3>=0:
            prev_round=game_context[n_match][n_round-3]
            start_ = prev_round.iloc[-1].ms-5000
            df = prev_round.query('ms>=@start_ and ms<=@start_+3000')
        elif start_time==3000 and n_round-3>=0:
            prev_round=game_context[n_match][n_round-3]
            start_ = game_context[n_match][n_round-3].iloc[-1].ms-2000
            df_1 = prev_round.query('ms>=@start_')
            df_2 = game_context[n_match][n_round-2].query('ms<=1000')
            df = df_1.append(df_2, ignore_index=True)
        elif start_time==3000 and n_round-3<0:
            df = game_context[n_match][n_round-2].query('ms<=1000')
        elif start_time-5000 >= end_time:
            df = game_context[n_match][n_round-2].query('ms>=@end_time-6000')   
        else:
            df = game_context[n_match][n_round-2].query('ms>=@start_time-5000 and ms<=@start_time-2000')
            
        #check events
        #self player
        if df.query('event=="player_hurt" and  users_self == @n_player').values.size:
            context_vector[0]=1
        if df.query('event=="player_death" and  users_self == @n_player').values.size:
            context_vector[1]=1
        if df.query('event=="player_blind" and  users_self == @n_player').values.size:
            context_vector[2]=1
        if df.query('event in @bomb_interaction_events and  users_self == @n_player').values.size:
            context_vector[3]=1
            
        #teammate player    
        if df.query('event=="player_hurt" and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[4]=1
        if df.query('event=="player_death" and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[5]=1
        if df.query('event in @bomb_interaction_events and  users_self != @n_player and ((users_self < 5 and @n_player < 5) or (users_self >= 5 and @n_player >= 5))').values.size:
            context_vector[6]=1
            
        #enemy player killed/hurted by player
        if df.query('event=="player_hurt" and  users_self != @n_player and users_attacker == @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[7]=1
        if df.query('event=="player_death" and  users_self != @n_player and users_attacker == @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[8]=1
            
        #enemy player killed/hurted by another player
        if df.query('event=="player_hurt" and  users_self != @n_player and users_attacker != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[9]=1
        if df.query('event=="player_death" and  users_self != @n_player and users_attacker != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[10]=1
        
        #enemy player interacted with bomb
        if df.query('event in @bomb_interaction_events and  users_self != @n_player and ((users_self < 5 and @n_player >= 5) or (users_self >= 5 and @n_player < 5))').values.size:
            context_vector[11]=1
            
    return context_vector



class CustomDataset(Dataset):

    def __init__(self, data_list,game_context):
        self.data_list = data_list
        self.game_context = game_context
        self.sampling_rate = 22050*3
    def __len__(self):
        return len(self.data_list)
  
    def __getitem__(self, index):
        item=self.data_list[index]
        sound_1d_array,_ = librosa.load(item[0]) # load audio to 1d array
        if sound_1d_array.size<self.sampling_rate:
            offset = self.sampling_rate - sound_1d_array.size 
            sound_1d_array = np.pad(sound_1d_array, (0, offset))
                
        if self.game_context is not None:
            ctx=get_context_vector(item[2],self.game_context)
            x=torch.from_numpy(np.concatenate([sound_1d_array,ctx],axis=0)).unsqueeze(0)
        else:
            x=torch.from_numpy(sound_1d_array).unsqueeze(0)
                
        y = create_onehot_tensor(item[1])

        return x,y


def prepare_data(file_path,path_to_audio,path_to_splitted_audio,test_size):
    print('Start splitting audio to ',path_to_splitted_audio)
    if not os.path.exists(path_to_splitted_audio):
        os.makedirs(path_to_splitted_audio)
    split_audio(path_to_audio,path_to_splitted_audio)
    print('Finish splitting\n')
    full_dict_with_emt = convert_dict(get_dict_with_emotions(file_path))
    print('Emotion statistic:')
    display_emt(full_dict_with_emt)
    print('\nPrepare train and val lists')
    start = time.time()
    train_list, val_list = split_annotations(full_dict_with_emt,path_to_splitted_audio,test_size)
    print("It took: ", round((time.time()-start)/60,2)," minutes")
    print ('train size: ', len(train_list))
    print ('val size: ', len(val_list))
    return train_list, val_list



def get_dataloader(file_path,path_to_audio,path_to_splitted_audio,test_size,use_game_context=False,batch_size=32):
  
    train_list, val_list = prepare_data(file_path,path_to_audio,path_to_splitted_audio,test_size)
    if use_game_context:
        game_context=get_game_context()
    else:
        game_context = None
    print('\nPrepate train dataset')
    train_dataset = CustomDataset(train_list,game_context=game_context)
    print('Prepate val dataset')
    val_dataset = CustomDataset(val_list,game_context=game_context)

    train_dataloader=DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=NUM_WORKERS, shuffle=True,pin_memory=True)

    val_dataloader=DataLoader(
                val_dataset, batch_size=batch_size,
                num_workers=NUM_WORKERS, shuffle=False,pin_memory=True)

    return train_dataloader,val_dataloader

#for ML methods
def get_data(data_list,game_context=False):

    sampling_rate = 22050*3
        
    x_tensors = []
    y_tensors = []
        
    for item in data_list:
            
            sound_1d_array,_ = librosa.load(item[0])
            if sound_1d_array.size<sampling_rate:
                offset = sampling_rate - sound_1d_array.size 
                sound_1d_array = np.pad(sound_1d_array, (0, offset))
                
            if game_context:
                ctx=get_context_vector(item[2])
                x_tensors.append(np.concatenate([sound_1d_array,ctx],axis=0))
            else:
                x_tensors.append(sound_1d_array)
                
            y_onehot = create_onehot_tensor(item[1]).numpy()
            y_tensors.append(y_onehot)

    return np.array(x_tensors), np.array(y_tensors)

def get_train_test(file_path,path_to_audio,path_to_splitted_audio,test_size,use_game_context=False):
    
    train_list, val_list = prepare_data(file_path,path_to_audio,path_to_splitted_audio,test_size)
    if use_game_context:
        game_context=get_game_context()
    else:
        game_context = None
    print('Prepate train dataset')
    x_train,y_train = get_data(train_list,game_context=game_context)
    print('Prepate test dataset')
    x_test,y_test = get_data(val_list,game_context=game_context)
    return x_train,y_train,x_test,y_test





