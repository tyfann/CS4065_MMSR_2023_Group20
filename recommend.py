import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('spotify_dataset_160k_tracks.csv.zip', compression='zip')
data.drop_duplicates(inplace=True, subset=['name'])
name = data['name']

col_features = ['danceability', 'energy', 'valence', 'loudness']
X = MinMaxScaler().fit_transform(data[col_features])
kmeans = KMeans(init="k-means++", n_clusters=2, random_state=15, n_init=10).fit(X)
data['kmeans'] = kmeans.labels_

data2 = data[:data.shape[0]]
cluster = data2.groupby(by=data2['kmeans'])
data2.pop('kmeans')
df1 = cluster.apply(lambda x: x.sort_values(["popularity"], ascending=False))
df1.reset_index(level=0, inplace=True)


def _get_results(emotion_code, last_song):
    NUM_RECOMMEND = 10
    happy_set = []
    sad_set = []
    happy_set2 = []
    sad_set2 = []
    if emotion_code == 0:
        #happy_set.append(df1[df1['kmeans'] == 0]['name'].head(NUM_RECOMMEND))
        #happy_set.append(df1[df1['kmeans'] == 0]['name'])
        happy_set.append(df1[df1['kmeans'] == 0])
        filtered_songs = happy_set[0]
    else:
        #sad_set.append(df1[df1['kmeans'] == 1]['name'].head(NUM_RECOMMEND))
        #sad_set.append(df1[df1['kmeans'] == 1]['name'])
        sad_set.append(df1[df1['kmeans'] == 1])
        filtered_songs = sad_set[0]

    # print(filtered_songs)
    # print(df1[df1['name'] == last_song])
    # print(cluster)

    if last_song in data['name'].values:
        similarity_scores = cosine_similarity(df1[df1['name'] == last_song][col_features], filtered_songs[col_features])
        similarity_scores = similarity_scores.flatten()
        similar_songs_indices = similarity_scores.argsort()[-NUM_RECOMMEND:][::-1]
        similar_songs = filtered_songs.iloc[similar_songs_indices]['name']
        filtered_songs = similar_songs
    else:
        if emotion_code == 0:
            happy_set2.append(df1[df1['kmeans'] == 0]['name'].head(NUM_RECOMMEND))
            return pd.DataFrame(happy_set2).T
        elif emotion_code == 1:
            sad_set2.append(df1[df1['kmeans'] == 1]['name'].head(NUM_RECOMMEND))
            return pd.DataFrame(sad_set2).T

    return pd.DataFrame(filtered_songs)


emotion_word = input("Are you in a good mood today?\n")
if emotion_word.lower() == 'no':
    emotion_code = 0
else:
    emotion_code = 1

last_song = input("What is the name of the last song you listened to?\n")

print(_get_results(emotion_code, last_song))
