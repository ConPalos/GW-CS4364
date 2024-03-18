import pandas as pd
import numpy as np
import re

def shape_data():
    '''
    Read and parse Spotify data from Kaggle
    @return: A pandas DataFrame with the data
    '''
    # read in the data
    try:
        data = pd.read_csv('spotify-2023.csv', encoding='latin-1')

    except:
        data = pd.read_csv('hw1/spotify-2023.csv', encoding='latin-1')

    # each column is going to handle being null differently
    
    # target will be in_spotify_playlists (how many times was this song added to a playlist)
    # this will be based on artist count, year, month, day, bpm, mode, and the percents
    # so start by filtering down to just those columns (plus the song name and artist(s) for human readability)
    data = data[[
        'track_name',           # nothing
        'artist(s)_name',       # nothing
        'artist_count',         # nothing
        'released_year',        # nothing
        'released_month',       # nothing
        'released_day',         # nothing
        'in_spotify_playlists', # drop
        'streams',              # drop
        'bpm',                  # nothing
        'mode',                 # nothing
        'danceability_%',       # nothing
        'valence_%',            # nothing
        'energy_%',             # drop
        'acousticness_%',       # nothing
        'instrumentalness_%',   # nothing
        'liveness_%',           # nothing
        'speechiness_%'         # nothing
    ]]

    # each column is commented with what to do in the event of a null value, so do it
    mask = data['streams'].isna()
    data = data[~mask]    
    mask = data['in_spotify_playlists'].isna()
    data = data[~mask]
    mask = data['energy_%'].isna()
    data = data[~mask]

    # # ok that's it for dropping data, now time to convert to raw numbers
    # mode_map = {
    #     'Major': 1,
    #     'Minor': -1
    # }
    # data['mode'] = data['mode'].apply(lambda row: mode_map[row])
    
    # if not an int, parse an int from it
    data['streams'] = data['streams'].apply(
        lambda row: extract_number(row)
    )
    # if nan, drop entirely
    data = data.dropna(subset='streams')
    data['streams'] = data['streams'].astype(int)
    # and now divide by one million to stop an overflow error (units become millions)
    data['streams'] /= 1000000

    # units become thousands
    data['in_spotify_playlists'] /= 1000

    # just return the data
    return data

def extract_number(string: str):
    if type(string) == str:
        result = re.findall(r'^\d+', string)
        if len(result) > 0:
            return result[0]
        
        return np.nan
    
    elif type(string) == int:
        return string
    
    return np.nan

def get_sample_data(size):
    data = np.random.rand(size, 2)
    randoms = np.random.rand(size)/2 - 0.5

    # for each row
    for row in range(len(data)):
        # the second column is a function: 2x^2 - 3x + err
        data[row, 1] = 2*(data[row, 0]**2) - 3*data[row, 0] + randoms[row]

    train_cutoff = int(np.ceil(size/5))
    test_cutoff = int(np.ceil(2*size/5))
    
    train = data[0:train_cutoff]
    test = data[train_cutoff:test_cutoff]
    holdout = data[test_cutoff:]

    return train, test, holdout

def get_sample_data_3d(size):
    data = np.random.rand(size, 3)
    randoms = np.random.rand(size)/2 - 0.5

    # for each row
    for row in range(len(data)):
        # the third column is a function: 2x^2 - 3x + y^2 + 2y + err
        data[row, 2] = 2*(data[row, 0]**2) - 3*data[row, 0] + data[row, 1]**2 + 2*data[row, 1] + randoms[row]

    train_cutoff = int(np.ceil(size/5))
    test_cutoff = int(np.ceil(2*size/5))
    
    train = data[0:train_cutoff]
    test = data[train_cutoff:test_cutoff]
    holdout = data[test_cutoff:]

    return train, test, holdout