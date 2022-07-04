import glob
import json
import os
import sys
import pathlib
import platform
from datetime import datetime
import time

from functools import lru_cache
from collections import Counter
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import seaborn as sns

import librosa
from pydub import AudioSegment

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from csgo_tournament_msu_labelled.csgo_tournament_metadata import *
from .csgo_dem_info_parsing import get_context_vector

path_delim = '\\' if platform.system() == 'Windows' else '/'
sys.path.insert(0, '..')
sys.path.insert(0, '../..')


# разбиение всех аудио по 3 сек в соответствии с csv разметкой
def split_audio(path_to_audio, path_to_splitted_audio):
    for path in pathlib.Path(path_to_audio).iterdir():
        if path.is_dir():
            player_number = str(path)[str(path).rfind(path_delim) + 1:].split('_')[0]
            for path_in in pathlib.Path(path).iterdir():
                str_path = str(path_in)
                name = str_path[str_path.rfind(path_delim) + 1:]
                sound = AudioSegment.from_wav(path_in)
                i = 0
                while i <= len(sound):
                    if i + 3000 > len(sound):
                        cut = sound[i:len(sound) + 1]
                        if not os.path.exists(path_to_splitted_audio +
                                              path_delim +
                                              player_number + '_' +
                                              name[:-4] + f'_{i}_{len(sound)}.wav'):
                            cut.export(path_to_splitted_audio +
                                       path_delim +
                                       player_number + '_' +
                                       name[:-4] + f'_{i}_{len(sound)}.wav',
                                       format="wav")
                        break
                    cut = sound[i:i + 3000]
                    if not os.path.exists(path_to_splitted_audio +
                                          path_delim +
                                          player_number + '_' + name[:-4] + f'_{i}_{i + 3000}.wav'):
                        cut.export(path_to_splitted_audio +
                                   path_delim +
                                   player_number + '_' +
                                   name[:-4] + f'_{i}_{i + 3000}.wav',
                                   format="wav")
                    i += 3000


# ## DATA HANDLING
@lru_cache(None)
def get_dict_with_emotions(file_path):  # keys: match -> n_round -> num_player value: dataframe
    all_emt_dict = {}
    col_emt_names = ["start", "end", "emt_est_1", "str_emt_est_1", "emt_est_2", "str_emt_est_2", "emt_est_3",
                     "str_emt_est_3"]
    for n_match, rounds_in_match in MATCH_LEN.items():
        match_emt_cl = {}
        for n_round in range(1, rounds_in_match + 1):
            players_emt = {}
            res_emt = {}
            for num_player in range(NUMBER_OF_PLAYERS):
                if glob.glob(file_path + path_delim + f'{num_player}_match{n_match}_round{n_round}.csv'):
                    str_path = glob.glob(file_path + path_delim + f'{num_player}_match{n_match}_round{n_round}.csv')[0]
                    try:
                        df_s = pd.read_csv(str_path, names=col_emt_names)
                    except UnicodeDecodeError:
                        if platform.system() == 'Windows':
                            os.system("notepad " + str_path)
                        else:
                            os.system("nano " + str_path)
                        df_s = pd.read_csv(str_path, names=col_emt_names)
                    res_emt[num_player] = df_s

            match_emt_cl[n_round] = res_emt
        all_emt_dict[n_match] = match_emt_cl
    return all_emt_dict


def find_majority(est_with_str):
    votes = [i[0] for i in est_with_str]
    strength = [i[1] for i in est_with_str]
    vote_count = Counter(votes)
    most_ = vote_count.most_common(1)
    if (most_[0][1] >= 2) and (most_[0][0] > 0):
        ids = [ind for ind in range(len(votes)) if votes[ind] == most_[0][0]]
        mean_str = round(np.array([strength[i] for i in ids]).mean(), 1)
        return [most_[0][0], mean_str]
    else:
        if (most_[0][0] == 0) and (most_[0][1] == 2):
            return [max(votes), max(strength)]
        return [-1, -1]


def get_emotions(df):
    player_emt = df.copy()
    for i in range(1, 4):
        player_emt[f'est_{i}'] = list(zip(player_emt[f'emt_est_{i}'], player_emt[f'str_emt_est_{i}']))

    ss = []
    for i in player_emt[['est_1', 'est_2', 'est_3']].values:
        ss.append(find_majority(i))
    player_emt['emt'] = np.asarray(ss)[:, 0]
    player_emt['emt'] = player_emt['emt'].apply(int)
    player_emt['str_emt'] = np.asarray(ss)[:, 1]

    emt_in_time = {}
    for i in player_emt[['start', 'emt']].query('emt>0').emt.unique():
        emt_in_time[i] = [j[0] for j in player_emt[['start', 'emt']].query('emt>0').values if j[1] == i]
    return emt_in_time


def convert_dict(all_emt_dict):
    full_emt = {}  # key - emotion_type: value - [n_player,n_match,n_round,start_time]
    for key in EMOTION_WITH_KEYS.keys():
        full_emt[key] = []

    for n_match, rounds_in_match in MATCH_LEN.items():
        for n_round in range(1, rounds_in_match + 1):
            for n_player in range(NUMBER_OF_PLAYERS):
                if all_emt_dict.get(n_match) is not None and \
                        all_emt_dict.get(n_match).get(n_round) is not None and \
                        all_emt_dict.get(n_match).get(n_round).get(n_player) is not None:
                    emt_in_time = get_emotions(all_emt_dict.get(n_match).get(n_round).get(n_player))
                    for e, start_time in emt_in_time.items():
                        full_emt[e].append([n_player, n_match, n_round, start_time])
    return full_emt


def split_annotations(full_emt, path_to_audio, test_size=0.2):
    train_annotations_list = []
    val_annotations_list = []

    X = []
    y = []
    info = []
    for key_emt, values in tqdm(full_emt.items()):
        for vals in tqdm(values):
            n_player, n_match, n_round, list_start_time = vals
            for start_time in list_start_time:
                file = glob.glob(
                    path_to_audio + path_delim + f'{n_player}_match{n_match}_round{n_round}_{start_time}*.wav')
                X.append(file[0])
                y.append(key_emt)
                info.append([n_player, n_match, n_round, start_time])

    #  не учитываем "стыд"
    #  слишком мало датапоинтов
    ind_to_del = y.index(9)
    y.pop(ind_to_del)
    X.pop(ind_to_del)
    info.pop(ind_to_del)

    X_train, X_val, y_train, y_val, info_train, info_val = train_test_split(X, y, info, test_size=test_size,
                                                                            random_state=42, shuffle=True, stratify=y)
    train_annotations_list = list(zip(X_train, y_train, info_train))
    val_annotations_list = list(zip(X_val, y_val, info_val))

    return train_annotations_list, val_annotations_list


def display_emt(full_dict_with_emt):
    for key, value in full_dict_with_emt.items():
        print(key, '-', np.array([len(i[3]) for i in value]).sum(), '   ', EMOTION_WITH_KEYS[key])


def create_onehot_tensor(label):
    y_onehot = torch.zeros(len(EMOTION_WITH_KEYS))
    y_onehot[label - 1] = 1
    return y_onehot

# ## DATA SET AND DATA LOADER OBJECTS
class BaseAudioSignalDataset(Dataset):
    def __init__(self, data_list, use_game_context, path_to_processed_csgo_data, sampling_rate=SAMPLING_RATE, **ignored_kwargs):
        self.data_list = data_list
        self.use_game_context = bool(use_game_context)
        self.sampling_rate = sampling_rate
        self.path_to_processed_csgo_data = path_to_processed_csgo_data

        if self.use_game_context:
            print("Preparing game context vectors")
            start = time.time()
            for _, _, info in tqdm(self.data_list):
                contexts = get_context_vector(*info, self.path_to_processed_csgo_data)
            print(f"Finished in {(time.time() - start) / 60:.2f} minutes")

    def __len__(self):
        return len(self.data_list)

    def load_wav(self, path):
        sound_1d_array, sr = librosa.load(path, sr=self.sampling_rate)  # load audio to 1d array
        if sound_1d_array.shape[-1] < sr * PCS_LEN_SEC:
            offset = sr * PCS_LEN_SEC - sound_1d_array.shape[-1]
            sound_1d_array = np.pad(sound_1d_array, (0, offset))
        return sound_1d_array

    def extract_features(self, path):
        x = self.load_wav(path)
        return torch.tensor(x)

    def __getitem__(self, index):
        item = self.data_list[index]
        x = self.extract_features(item[0])
        y = item[1] - 1

        if self.use_game_context:
            ctx = torch.from_numpy(get_context_vector(*item[2], self.path_to_processed_csgo_data)).float()
        else:
            ctx = torch.tensor([])
        return x, ctx, y


class BaseSpectrogramDataset(BaseAudioSignalDataset):
    def __init__(self,
                 data_list,
                 path_to_processed_csgo_data,
                 use_game_context=True,
                 sampling_rate=SAMPLING_RATE,
                 window_size=512,
                 **ignored_kwargs):
        self.window_size = window_size
        self.hop_len = self.window_size // 2
        self.path_to_processed_csgo_data = path_to_processed_csgo_data

        self.window_weights = np.hanning(self.window_size)[:, None]
        super().__init__(data_list, use_game_context, self.path_to_processed_csgo_data, sampling_rate)

    @staticmethod
    def __visualize__(spec):
        ax = sns.heatmap(spec)
        ax.invert_yaxis()

    def extract_features(self, path):
        _track = self.load_wav(path)

        spec = self.calculate_all_windows(_track)
        spec_offset = spec - np.min(spec)
        spec_offset = 255 * spec_offset/np.max(spec_offset)
        spec_offset = np.round(spec_offset)

        channel_amt = 13 if self.use_game_context else 1
        out = torch.zeros(channel_amt, *spec.shape)

        # out[0] = torch.tensor(spec_offset, dtype=torch.uint8)
        out[0] = torch.tensor(spec_offset)
        return out

    def __getitem__(self, index):
        item = self.data_list[index]
        x = self.extract_features(item[0])
        y = item[1] - 1

        if self.use_game_context:
            ctx = get_context_vector(*item[2], self.path_to_processed_csgo_data)
            for event, val in enumerate(ctx):
                x[event+1] = 255 * torch.ones_like(x[0]) if val else torch.zeros_like(x[0])
        else:
            ctx = torch.tensor([])
        return x, ctx, y

    """
    For a typical speech recognition task, 
    a window of 20 to 30ms long is recommended.
    The overlap can vary from 25% to 75%.
    it is kept 50% for speech recognition.
    """

    def calculate_all_windows(self, audio):

        truncate_size = (len(audio) - self.window_size) % self.hop_len
        audio = audio[:len(audio) - truncate_size]

        nshape = (self.window_size, (len(audio) - self.window_size) // self.hop_len)
        nhops = (audio.strides[0], audio.strides[0] * self.hop_len)

        windows = np.lib.stride_tricks.as_strided(audio,
                                                  shape=nshape,
                                                  strides=nhops)

        assert np.all(windows[:, 1] == audio[self.hop_len:(self.hop_len + self.window_size)])

        yf = np.fft.rfft(windows * self.window_weights, axis=0)
        yf = np.abs(yf) ** 2

        scaling_factor = np.sum(self.window_weights ** 2) * self.sampling_rate
        yf[1:-1, :] *= (2. / scaling_factor)
        yf[(0, -1), :] /= scaling_factor

        xf = float(self.sampling_rate) / self.window_size * np.arange(yf.shape[0])

        indices = np.where(xf <= self.sampling_rate // 2)[0][-1]
        return np.log(yf[:indices, :] + 1e-16)


class NpEncoder(json.JSONEncoder):
    """
    support class to reinforce JSON.dump handling as it's unfamiliar with np data types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def prepare_data(file_path, path_to_audio, path_to_splitted_audio, test_size):
    if not os.path.exists(path_to_splitted_audio):
        print('Start splitting audio to ', path_to_splitted_audio)
        os.makedirs(path_to_splitted_audio)
        split_audio(path_to_audio, path_to_splitted_audio)
        print('Finish splitting\n')
    else:
        print(f"{path_to_splitted_audio} exists; skip splitting")
    try:
        with open("prepared.json", "r") as file:
            train_list, val_list = json.load(file)
    except:
        full_dict_with_emt = convert_dict(get_dict_with_emotions(file_path))
        print('Emotion statistic:')
        display_emt(full_dict_with_emt)
        print('\nPrepare train and val lists')
        start = time.time()
        train_list, val_list = split_annotations(full_dict_with_emt, path_to_splitted_audio, test_size)
        print("It took: ", round((time.time() - start) / 60, 2), " minutes")
        print('train size: ', len(train_list))
        print('val size: ', len(val_list))

        if os.path.exists("prepared.json"):
            print("json already exists")
        else:
            with open("prepared.json", "a+") as file:
                json.dump([train_list, val_list], file, cls=NpEncoder)
    return train_list, val_list


def get_dataloader(file_path, path_to_audio, path_to_splitted_audio, path_to_processed_csgo_data,
                   test_size,
                   use_game_context=False,
                   batch_size=32,
                   DatasetClass=BaseAudioSignalDataset,
                   num_workers=1,
                   train_list=None,
                   val_list=None):
    if train_list is None:
        val_list = None
        train_list, val_list = prepare_data(file_path, path_to_audio, path_to_splitted_audio, test_size)

    print('\nPrepare train dataset')
    train_dataset = DatasetClass(train_list, path_to_processed_csgo_data=path_to_processed_csgo_data,
                                 use_game_context=use_game_context, num_workers=num_workers)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=True, pin_memory=True)

    print('Prepare val dataset')
    val_dataset = DatasetClass(val_list, path_to_processed_csgo_data=path_to_processed_csgo_data,
                               use_game_context=use_game_context, num_workers=num_workers)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


# for ML methods
def get_data(data_list, use_game_context, path_to_processed_csgo_data):
    amt_of_samples = SAMPLING_RATE * PCS_LEN_SEC

    X = []
    Y = []
    ctx_ = []

    for item in data_list:

        sound_1d_array, _ = librosa.load(item[0])
        if sound_1d_array.size < amt_of_samples:
            offset = amt_of_samples - sound_1d_array.size
            sound_1d_array = np.pad(sound_1d_array, (0, offset))

        X.append(sound_1d_array)
        y_onehot = create_onehot_tensor(item[1]).numpy()
        Y.append(y_onehot)
        if use_game_context:
            ctx_.append(get_context_vector(*item[2], path_to_processed_csgo_data))

    if use_game_context:
        return np.array(X), np.array(ctx_), np.array(Y)

    return np.array(X), np.array(Y)


def get_train_test(file_path, path_to_audio, path_to_splitted_audio, test_size, use_game_context=False):
    train_list, val_list = prepare_data(file_path, path_to_audio, path_to_splitted_audio, test_size)
    if use_game_context:
        print('Prepate train dataset')
        x_train, ctx_train, y_train = get_data(train_list, use_game_context=use_game_context)
        print('Prepate test dataset')
        x_test, ctx_test, y_test = get_data(val_list, use_game_context=use_game_context)
        return x_train, ctx_train, y_train, x_test, ctx_test, y_test
    else:
        print('Prepate train dataset')
        x_train, y_train = get_data(train_list, use_game_context=use_game_context)
        print('Prepate test dataset')
        x_test, y_test = get_data(val_list, use_game_context=use_game_context)
        return x_train, y_train, x_test, y_test
