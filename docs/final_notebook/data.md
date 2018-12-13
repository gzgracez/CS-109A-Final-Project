---
title: Data
layout: default
nav_order: 3
---


**Harvard University**<br/>
**Fall 2018**<br/>
**Instructors**: Pavlos Protopapas, Kevin Rader<br/>
**Group Number**: 49<br/>
**Group Members**: Tejal Patwardhan, Akshitha Ramachandran, Grace Zhang
<hr style="height:2pt">


```python
#RUN THIS CELL 
import requests
from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
```




<style>
blockquote { background: #AEDE94; }
h1 { 
    padding-top: 25px;
    padding-bottom: 25px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
h2 { 
    padding-top: 10px;
    padding-bottom: 10px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}

div.exercise {
	background-color: #ffcccc;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
}
div.theme {
	background-color: #DDDDDD;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 18pt;
}
div.gc { 
	background-color: #AEDE94;
	border-color: #E9967A; 	 
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 12pt;
}
p.q1 { 
    padding-top: 5px;
    padding-bottom: 5px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}
header {
   padding-top: 35px;
    padding-bottom: 35px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
</style>






```python
# import necessary notebooks
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.api import OLS

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix
import seaborn as sns
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold

sns.set(style='whitegrid')
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)

import random

%matplotlib inline
```

<hr style="height:2pt">

# Data Collection and Cleaning

We collected our data by using the Spotify API to create a json file of tracks and their features. Additionally, we labeled each track with a new feature , `in_playlist`, which equals 1 if Grace would include the track in her playlist and 0 if Grace would not include the track in her playlist. 

We accomplished this by manually creating 2 separate playlists, where one playlist includes random songs that Grace would include in her playlist and the other playlist includes random songs that Grace would not include in her playlist. We used the Spotify API `user_playlist_tracks` endpoint to collect some features, including `track_id`s, of the tracks in each of these playlists. We then used the `audio_features` endpoint of the Spotify API to get additional features like `danceability`, etc. for each of our tracks. Finally, we added the `in_playlist` feature to each of our tracks and wrote our final object to `spotify.json`.


<hr style="height:2pt">

# Data Description
Our data includes the following features:
- `danceability`: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
- `energy`: Energy represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. A value of 0.0 is least energetic and 1.0 is most energetic. 
- `key`: The estimated overall key of the track. Integers map to pitches using standard Pitch Class Notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- `loudness`: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values range between -60 and 0 db. 
- `mode`: Mode represents the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Mode is binary; major is represented by 1 and minor is 0.
- `speechiness`: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- `acousticness`: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- `liveness`: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. 
- `valence`: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- `tempo`: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 
- `duration_ms`: The duration of the track in milliseconds.
- `time_signature`: An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
- `popularity`: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. 
- `in_playlist`: Response variable. Categorical variable for whether in playlist of desire. 1 if in playlist, 0 if not in playlist.

The following features were recorded to help with visualization later, but not used as predictors in our analysis, as they are not characteristics of the music itself.
- `name`: Song title
- `artist`: First artist of song
- `type`: The object type: “audio_features”
- `id`: The Spotify ID for the track.
- `uri`: The Spotify URI for the track.
- `track_href`: A link to the Web API endpoint providing full details of the track.
- `analysis_url`: An HTTP URL to access the full audio analysis of this track. An access token is required to access this data.

# Exploratory Data Analysis


```python
random.seed(1)
```


```python
# load in dataset
spotify_df = pd.read_csv("data/spotify-more2.csv")

# drop unnecessary columns
spotify_df = spotify_df.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url', 'name', 'artist', 'Unnamed: 0'])
```


```python
# display head of data
display(spotify_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>in_playlist</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>popularity</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.929</td>
      <td>0.516</td>
      <td>138760</td>
      <td>0.0663</td>
      <td>0</td>
      <td>0.000972</td>
      <td>7</td>
      <td>0.1120</td>
      <td>-19.221</td>
      <td>0</td>
      <td>11</td>
      <td>0.0334</td>
      <td>109.879</td>
      <td>4</td>
      <td>0.278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.539</td>
      <td>0.454</td>
      <td>324133</td>
      <td>0.2600</td>
      <td>0</td>
      <td>0.000780</td>
      <td>8</td>
      <td>0.0675</td>
      <td>-13.193</td>
      <td>0</td>
      <td>63</td>
      <td>0.0401</td>
      <td>174.322</td>
      <td>5</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.360</td>
      <td>0.676</td>
      <td>205773</td>
      <td>0.4400</td>
      <td>0</td>
      <td>0.000069</td>
      <td>0</td>
      <td>0.1620</td>
      <td>-11.960</td>
      <td>1</td>
      <td>59</td>
      <td>0.0291</td>
      <td>80.434</td>
      <td>4</td>
      <td>0.499</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.984</td>
      <td>0.466</td>
      <td>294307</td>
      <td>0.0718</td>
      <td>0</td>
      <td>0.000931</td>
      <td>0</td>
      <td>0.1070</td>
      <td>-17.999</td>
      <td>1</td>
      <td>56</td>
      <td>0.0374</td>
      <td>121.885</td>
      <td>4</td>
      <td>0.196</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.779</td>
      <td>0.496</td>
      <td>423573</td>
      <td>0.6340</td>
      <td>0</td>
      <td>0.402000</td>
      <td>5</td>
      <td>0.0746</td>
      <td>-10.328</td>
      <td>0</td>
      <td>60</td>
      <td>0.0364</td>
      <td>93.357</td>
      <td>4</td>
      <td>0.606</td>
    </tr>
  </tbody>
</table>
</div>



```python
# display shape of data
display(spotify_df[spotify_df["in_playlist"]==0].shape)
```


    (2500, 15)


We have 5060 songs in our initial analysis. 2650 are included in Grace's playlist, and 2500 are not included in Grace's playlist. 


```python
# generate summary chart of features
features = []
means = []
var = []
ranges = []
mins = []
maxes = []

for feature in spotify_df:
    if feature != "in_playlist":
        features.append(feature)
        means.append(spotify_df[feature].mean())
        var.append(spotify_df[feature].var())
        ranges.append(spotify_df[feature].ptp())
        mins.append(spotify_df[feature].min())
        maxes.append(spotify_df[feature].max())

summary_df = pd.DataFrame(data = {'feature': features, 
                                  'mean': means,
                                  'var' : var,
                                  'range': ranges, 
                                  'min': mins, 
                                  'max': maxes}) 
```

Below are summary statistics for all the features we plan to analyze:


```python
display(summary_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>mean</th>
      <th>var</th>
      <th>range</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>acousticness</td>
      <td>0.540199</td>
      <td>1.267884e-01</td>
      <td>9.959953e-01</td>
      <td>0.000005</td>
      <td>0.996</td>
    </tr>
    <tr>
      <th>1</th>
      <td>danceability</td>
      <td>0.570920</td>
      <td>2.931912e-02</td>
      <td>9.162000e-01</td>
      <td>0.061800</td>
      <td>0.978</td>
    </tr>
    <tr>
      <th>2</th>
      <td>duration_ms</td>
      <td>245718.492885</td>
      <td>1.911563e+10</td>
      <td>3.346533e+06</td>
      <td>44507.000000</td>
      <td>3391040.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>energy</td>
      <td>0.439224</td>
      <td>6.633419e-02</td>
      <td>9.901450e-01</td>
      <td>0.000855</td>
      <td>0.991</td>
    </tr>
    <tr>
      <th>4</th>
      <td>instrumentalness</td>
      <td>0.143138</td>
      <td>9.302492e-02</td>
      <td>9.870000e-01</td>
      <td>0.000000</td>
      <td>0.987</td>
    </tr>
    <tr>
      <th>5</th>
      <td>key</td>
      <td>5.223913</td>
      <td>1.251578e+01</td>
      <td>1.100000e+01</td>
      <td>0.000000</td>
      <td>11.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>liveness</td>
      <td>0.163377</td>
      <td>1.798945e-02</td>
      <td>9.800000e-01</td>
      <td>0.012000</td>
      <td>0.992</td>
    </tr>
    <tr>
      <th>7</th>
      <td>loudness</td>
      <td>-10.270219</td>
      <td>3.464989e+01</td>
      <td>4.217600e+01</td>
      <td>-42.476000</td>
      <td>-0.300</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mode</td>
      <td>0.650198</td>
      <td>2.274856e-01</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>popularity</td>
      <td>36.977470</td>
      <td>4.773025e+02</td>
      <td>1.000000e+02</td>
      <td>0.000000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>speechiness</td>
      <td>0.070655</td>
      <td>6.217856e-03</td>
      <td>8.989000e-01</td>
      <td>0.023100</td>
      <td>0.922</td>
    </tr>
    <tr>
      <th>11</th>
      <td>tempo</td>
      <td>117.657563</td>
      <td>8.604272e+02</td>
      <td>1.790410e+02</td>
      <td>42.581000</td>
      <td>221.622</td>
    </tr>
    <tr>
      <th>12</th>
      <td>time_signature</td>
      <td>3.919763</td>
      <td>1.655315e-01</td>
      <td>5.000000e+00</td>
      <td>0.000000</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>valence</td>
      <td>0.425801</td>
      <td>5.455384e-02</td>
      <td>9.591000e-01</td>
      <td>0.025900</td>
      <td>0.985</td>
    </tr>
  </tbody>
</table>
</div>


We can see that all features have values that are expected as per the Spotify API documentation. To analyze each feature in more granularity we looked at density plots. 


```python
# define response column
response_col = 'in_playlist'
```


```python
# prepare data for display
resp_col_loc = list(spotify_df.columns).index('in_playlist')
spotify_graphs_df = spotify_df.drop(columns=[response_col])
num_cols = len(spotify_graphs_df.columns)
nbin = 15

# iterate through all the features and display them
fig, axs = plt.subplots(num_cols, 1, figsize=(10,50))
for i in range(num_cols):
    sns.distplot(spotify_graphs_df[spotify_df.in_playlist == 0][spotify_graphs_df.columns[i]], hist = False, kde = True, ax=axs[i])
    sns.distplot(spotify_graphs_df[spotify_df.in_playlist == 1][spotify_graphs_df.columns[i]], hist = False, kde = True, ax=axs[i])
    axs[i].set_title("Density of " + str(spotify_graphs_df.columns[i]))
    axs[i].set_ylabel(r'Frequency')
fig.subplots_adjust(hspace=.5)
plt.show()
```


![png](output_18_0.png)


Looking at the density plots above, we note some features that show clear differences in distribution between the playlist and non-playlist. 
While non-playlist songs contain a roughly uniform distribution of energy values, playlist songs spike at an energy level between 0.2-0.4.
Acousticness in playlist tracks is much higher on average, spiking around 0.8, while non-playlist tracks most frequently have acousticness values around 0.1.
Instrumentalness is a particularly interesting feature. While the distribution non-playlist tracks is bimodal, peaking at around 0 and 0.9, playlist tracks have a few very well-defined peaks between 0 and 0.3. 
We will note in advance that this may induce a risk of overfitting based on instrumentalness values.
Playlist tracks have lower loudnesses on average, centering around -10, while non-playlist tracks -5.
In terms of speechiness, the distribution for playlist tracks has a much lower variance and slightly lower expected value, centering around 0.3 while non-playlist tracks center around 0.4.
Valence for non-playlist tracks is roughly uniformly distributed, while playlist tracks demonstrate a roughly normal distribution centered around 0.3.
Finally in terms of popularity, playlist tracks show a peak in their distribution around 60, while non-playlist tracks have a more variable distribution with a peak between 45-55.
The rest of the features are roughly similar in distribution between playlist and non-playlist tracks.


```python
# pair plots
ax = sns.pairplot(spotify_df, hue = "in_playlist", diag_kind="kde")
ax
plt.show()
```

    /usr/share/anaconda3/lib/python3.6/site-packages/statsmodels/nonparametric/kde.py:488: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    /usr/share/anaconda3/lib/python3.6/site-packages/statsmodels/nonparametric/kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    /usr/share/anaconda3/lib/python3.6/site-packages/numpy/core/_methods.py:26: RuntimeWarning: invalid value encountered in reduce
      return umr_maximum(a, axis, None, out, keepdims)



![png](output_20_1.png)


The pairplot above demonstrates a few interesting things. First, we notice positive correlations between loudness and energy, loudness and danceability, and danceablility and loudness. All three correlations are weak to moderate. We also notice a negative correlation between acousticness and energy. 