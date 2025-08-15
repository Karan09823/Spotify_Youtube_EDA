# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded=files.upload()

df = pd.read_csv("SpotifyYoutubeDataset.csv")

df.head()

df.columns

df.info()

# normalize column name
df.columns=df.columns.str.strip().str.lower().str.replace(' ','_')
df.columns

# drop unwanted columns
df.drop(columns=['unnamed:_0','url_spotify','url_youtube'],inplace=True)

df.info()

df.isna().sum()

# filling missing values with 0 in like and comment column
df['likes'] = df['likes'].fillna(0)
df['comments'] = df['comments'].fillna(0)

df.isnull().sum()

df.dropna(inplace=True)

df.isna().sum()

df.info()

# Top 10 Artists - with the Highest Views on YouTube?
top_10_artists=df.groupby('artist')['views'].sum().sort_values(ascending=False).head(10)
print(top_10_artists)

plt.figure(figsize=(17, 3))
sns.barplot(x=top_10_artists.index,y=top_10_artists.values,hue=top_10_artists.index,legend=False,palette='viridis')
plt.xlabel('Artist')
plt.xticks(rotation=30)
plt.ylabel('Views')
plt.title('Top 10 Artists - with the highest views on Youtube')
plt.show()

# Top 10 Tracks - with the Highest Streams on Spotify?
top_10_tracks=df.groupby('track')['stream'].sum().sort_values(ascending=False).head(10)
print(top_10_tracks)

plt.figure(figsize=(20,4))
sns.barplot(x=top_10_tracks.index,y=top_10_tracks.values,hue=top_10_tracks.index,legend=False,palette='Set1')
plt.xlabel('Tracks')
plt.ylabel('Streams')
plt.xticks(rotation=30)
plt.title('Top 10 Tracks - with the highest Streams on Spotify')
plt.show()

#What are the most common Album Types on Spotify? How many tracks belong to each album type?
most_common_album_type=df['album_type'].value_counts()
print(most_common_album_type)

plt.pie(most_common_album_type,labels=most_common_album_type.index,autopct='%1.1f%%',startangle=90)
plt.title('Most Common Album Types on Spotify')
plt.show()

avg_views_likes_comments = df.groupby('album_type')[['views', 'likes', 'comments']].mean()
print(avg_views_likes_comments)


#Reshape DataFrame for grouped barplot
avg_df = avg_views_likes_comments.reset_index()

#melt
avg_melted = avg_df.melt(
    id_vars='album_type', # What to keep as-is (group column)
    var_name='Metric', # Name for the melted column headers
    value_name='Average') # Name for the values



plt.figure(figsize=(12, 5))
sns.barplot(data=avg_melted, x='album_type', y='Average', hue='Metric', palette='pastel')
plt.title('Average Views, Likes, and Comments by Album Type')
plt.xlabel('Album Type')
plt.ylabel('Average Value')
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

#Top 5 YouTube Channels -  based on the Views?
top_5_yt_channel=df.groupby('channel')['views'].sum().sort_values(ascending=False).head(5)
print(top_5_yt_channel)

plt.figure(figsize=(10,5))
sns.barplot(x=top_5_yt_channel.index,y=top_5_yt_channel.values,hue=top_5_yt_channel.index,legend=False,palette='viridis')
plt.xlabel('Channel')
plt.ylabel('Views')
plt.title('Top 5 Youtube Channels - based on the Views')
plt.show()

# The Top Most Track -  based on Views?
top_most_track=df.groupby('track')['views'].sum().sort_values(ascending=False).head()
print(top_most_track)

plt.figure(figsize=(14,4))
sns.barplot(x=top_most_track.index,y=top_most_track.values,hue=top_most_track.index,legend=False,palette='Set2')
plt.xlabel('Track')
plt.ylabel('Views')
plt.title('The Top Most Track -  based on Views')
plt.show()

# Which Top 7 Tracks have the highest Like-to-View ratio on YouTube?
df['likes_to_view_ratio']=df['likes']/df['views']

top_7_tracks=df.sort_values(by='likes_to_view_ratio',ascending=False).head(7)
#print(top_7_tracks)

plt.figure(figsize=(14,4))
sns.barplot(x=top_7_tracks['track'],y=top_7_tracks['likes_to_view_ratio'],hue=top_7_tracks['track'],legend=False,palette='Set2')
plt.title('Top 7 Tracks having the highest like-to-view ration on Youtube')
plt.xlabel('Track')
plt.xticks(rotation=20)
plt.ylabel('Likes to View Ratio')
plt.show()

# Top Albums having the Tracks with Maximum Danceability ?
max_dance_by_album=df.groupby('album')['danceability'].max().reset_index()
merged_df=pd.merge(df,max_dance_by_album,on=['album','danceability'],how='inner')
merged_df=merged_df[['album','track','danceability']].sort_values(by='danceability',ascending=False)
merged_df.head()

# What is the Correlation between Views, Likes, Comments, and Stream?
cols=['views','likes','comments','stream']
correlation_matrix=df[cols].corr()

sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidth=0.5)
plt.title('Correlation between Views, Likes, Comments, and Stream')
plt.show()

# Which artists have the most tracks in the dataset?
df_tracks=df.groupby('artist')['track'].count().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,5))
sns.barplot(x=df_tracks.values,y=df_tracks.index,palette='viridis')
plt.title('Top 10 artist with most tracks')
plt.xlabel('Number of Tracks')
plt.ylabel('Artist')
plt.show()

# Which albums appear most frequently?
df_most_appeared_album=df['album'].value_counts().head(20)

plt.figure(figsize=(12,5))
sns.barplot(x=df_most_appeared_album.values,y=df_most_appeared_album.index,palette='viridis')
plt.title('Top 20 most appeared albums')
plt.xlabel('Number of appearances')
plt.ylabel('Album')

# What’s the most common key or tempo range of songs?
most_common_key=df['key'].value_counts().idxmax()
most_common_tempo=df['tempo'].value_counts().idxmax()

print(f"Most common key: {most_common_key}")
print(f"Most common tempo: {most_common_tempo}")

plt.figure(figsize=(12, 5))

# Histogram for Key
sns.histplot(df['key'], kde=False, color='blue', bins=len(df['key'].unique()), label='Key')

# Histogram for Tempo
sns.histplot(df['tempo'], kde=True, color='red', label='Tempo', alpha=0.5)

plt.title('Distribution of Key and Tempo of Songs', fontsize=14, weight='bold')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Which songs have the longest and shortest durations?
df_longShortduration=df.groupby('track')['duration_ms'].sum().sort_values(ascending=False)

longest_track_name=df_longShortduration.index[0]
print(f"Longest song duration name: {longest_track_name}")
print()
shortest_track_name=df_longShortduration.index[-1]
print(f"Shortest song duration name: {shortest_track_name}")

# What’s the overall distribution of energy, danceability, valence?
features = ['energy', 'danceability', 'valence']


plt.figure(figsize=(12, 6))

# Histogram + KDE for each feature
for feature in features:
    sns.kdeplot(df[feature], shade=True, label=feature)

plt.title('Distribution of Energy, Danceability, and Valence', fontsize=14, weight='bold')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# Are there outliers in loudness or instrumentalness
def find_outlier(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

#
loudness_outliers = find_outlier(df['loudness'])
instrumentalness_outliers = find_outlier(df['instrumentalness'])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.boxplot(y=df['loudness'])
plt.title('Loudness')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['instrumentalness'])
plt.title('Instrumentalness')

plt.tight_layout()
plt.show()

# What’s the correlation between danceability, energy, and valence?
features = df[['danceability', 'energy', 'valence']]
correlation_matrix = features.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
plt.title('Correlation between Danceability, Energy, and Valence')
plt.show()

# Do songs with high energy tend to have lower acousticness?
plt.figure
sns.scatterplot(data=df,x='energy',y='acousticness')
plt.title('Do songs with high energy tend to have lower acousticness')
plt.xlabel('Energy')
plt.ylabel('Acousticness')
plt.show()

# Are instrumental songs more or less popular on YouTube?
df['instrumental_category'] = df['instrumentalness'].apply(lambda x: 'Instrumental' if x >= 0.5 else 'Non-instrumental')

# Create boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='instrumental_category', y='views')
plt.title('Instrumental vs Non-Instrumental Songs: YouTube Views')
plt.xlabel('Song Type')
plt.ylabel('YouTube Views')
plt.grid(True, axis='y')
plt.show()

# How does tempo vary by album type or artist?
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='album_type', y='tempo')
plt.title('Tempo Distribution by Album Type')
plt.xlabel('Album Type')
plt.ylabel('Tempo (BPM)')
plt.xticks(rotation=45)
plt.show()

# Which tracks have the highest views, likes, and comments? (Top 10)
df_topTracks=df.groupby('track')[['views','likes','comments']].sum().sort_values(by='views',ascending=False).head(10)
sns.barplot(data=df_topTracks,x='views',y=df_topTracks.index,hue='likes',palette='viridis')
plt.title('Top 10 Tracks with Highest Views, Likes, and Comments')
plt.xlabel('Views, Likes, and Comments')
plt.ylabel('Tracks')
plt.show()

# Are there artists whose videos consistently get high engagement?

# engagement rate= (like+comments)/views
df['engagement_rate']=(df['likes']+df['comments'])/df['views']
artist_engagement = df.groupby('artist')['engagement_rate'].agg(['mean', 'std', 'count'])
artist_engagement = artist_engagement[artist_engagement['count'] >= 5]
artist_engagement_sorted = artist_engagement.sort_values(by=['mean', 'std'], ascending=[False, True])

print(artist_engagement_sorted.head(10))

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=artist_engagement,
    x='std', y='mean', size='count', hue='mean', palette='viridis', legend=False
)
plt.title('Artist Engagement Consistency')
plt.xlabel('Engagement Rate Std Dev (Lower = More Consistent)')
plt.ylabel('Average Engagement Rate')
plt.show()

# Which channels upload the most official videos?
df_mostOfficial=df.groupby('channel')['official_video'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=df_mostOfficial.values,y=df_mostOfficial.index,palette='viridis')
plt.title('Top 10 Channels which uploads most official videos')
plt.xlabel('Number of Official Videos')
plt.ylabel('Channel')
plt.show()

#Do licensed tracks perform better than non-licensed ones?
sns.boxplot(data=df,x='licensed',y='views')
plt.title('Licensed vs Views')
plt.xlabel('Licensed')
plt.ylabel('Views')
plt.show()

# Do Spotify danceability and YouTube views show any correlation?
sns.scatterplot(data=df,x='danceability',y='views')
plt.title('Danceability vs Views')
plt.xlabel('Danceability')
plt.ylabel('Views')
plt.show()

# Does energy or valence predict YouTube popularity?
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='energy',
    y='valence',
    size='views',      # bubble size = popularity
    hue='views',       # color intensity = popularity
    palette='viridis',
    alpha=0.7,
    sizes=(20, 200)
)
plt.title('Energy & Valence vs YouTube Views')
plt.xlabel('Energy')
plt.ylabel('Valence')
plt.legend(title='Views', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Are high loudness songs more popular on YouTube?
sns.scatterplot(data=df,x='loudness',y='views')
plt.title('Loudness vs Views')
plt.xlabel('Loudness')
plt.ylabel('Views')
plt.show()

# Is there a trend between speechiness and popularity? (e.g., rap vs instrumental)
sns.scatterplot(data=df,x='speechiness',y='views')
plt.title('Speechiness vs Views')
plt.xlabel('Speechiness')
plt.ylabel('Views')
plt.show()

