# %%
import pandas as pd
import numpy as np
from data import shape_data
import pdb
from nonlinear import NLSModel
from linear import LinearRegressor
import matplotlib.pyplot as plt

#%%
# retrieve the data
train, test, holdout = shape_data()

# convert train data to numpy
cols = [
    # 'artist_count',
    # 'released_year',
    # 'released_month',
    # 'released_day',
    'in_spotify_playlists',
    # 'bpm',
    # 'mode',
    # 'danceability_%',
    # 'valence_%',
    # 'energy_%',
    # 'acousticness_%',
    # 'instrumentalness_%',
    # 'liveness_%',
    # 'speechiness_%'
]

train_inputs = train[cols].to_numpy()
train_answer = train['streams'].to_numpy()

#%%
# craft the linear model
linear = LinearRegressor(0.00000000001, 1, 100)

scores, weights = linear.fit(train_inputs, train_answer)
#%%
# craft the regression model
model = NLSModel(0.00000000001, 1, 100, 1)

# fit the data
model.fit(train_inputs, train_answer)
# %%