"""Prototype of mood-based recommendation engine."""
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('spotify_dataset_160k_tracks.csv.zip', compression='zip')
data.drop_duplicates(inplace=True, subset=['name'])
name = data['name']

col_features = ['danceability', 'energy', 'valence', 'loudness']
X = MinMaxScaler().fit_transform(data[col_features])
kmeans = KMeans(init="k-means++",
                n_clusters=2,
                random_state=15,
                n_init=10).fit(X)
data['kmeans'] = kmeans.labels_

data2 = data[:data.shape[0]]
cluster = data2.groupby(by=data2['kmeans'])
data2.pop('kmeans')
df1 = cluster.apply(lambda x: x.sort_values(["popularity"], ascending=False))
df1.reset_index(level=0, inplace=True)


def _get_results(emotion_code):
    NUM_RECOMMEND = 10
    happy_set = []
    sad_set = []
    if emotion_code == 0:
        happy_set.append(df1[df1['kmeans'] == 0]['name'].head(NUM_RECOMMEND))
        return pd.DataFrame(happy_set).T
    else:
        sad_set.append(df1[df1['kmeans'] == 1]['name'].head(NUM_RECOMMEND))
        return pd.DataFrame(sad_set).T


emotion_word = input("Are you in a good mood today?\n")
if emotion_word == 'no':
    emotion_code = 0
else:
    emotion_code = 1

print(_get_results(emotion_code))
