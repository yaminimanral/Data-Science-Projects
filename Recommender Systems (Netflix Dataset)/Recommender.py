import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import missingno as msno
# %matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.impute import SimpleImputer

# Importing the Netflix Titles data set into the DataFrame
netflix_titles = pd.read_csv("Netflix_Titles_Dataset.csv", encoding='utf-8')

# Resetting index for the DataFrame by dropping the Index column
merged_df = pd.concat([netflix_titles], axis = 0).reset_index()
merged_df = merged_df.drop(['index'], axis=1)

# Handling the Genres column to fetch the first Genre from the list of Genres
merged_df['genres'] = merged_df['genres'].str.replace(r'[','').str.replace(r"'",'').str.replace(r']','')
merged_df['genre'] = merged_df['genres'].str.split(',').str[0]

# Handling the Production Countries column to fetch the first Country from the list of Country
merged_df['production_countries'] = merged_df['production_countries'].str.replace(r"[", '').str.replace(r"'", '').str.replace(r"]", '')
merged_df['production_country'] = merged_df['production_countries'].str.split(',').str[0]

# Removing unwanted columns for further processing
merged_df = merged_df.drop(['genres', 'production_countries'], axis=1)

# Replacing the blank value with NULL in Genre and Production Country column
merged_df['genre'] = merged_df['genre'].replace('', np.nan)
merged_df['production_country'] = merged_df['production_country'].replace('',np.nan)

# Checking the data for the type 'Movie'
merged_df[merged_df['type'] == 'MOVIE']

# Setting the value for all the blank entries of seasons as 0
merged_df['seasons'] = merged_df['seasons'].fillna(0)

# Dropping unwanted columns not needed for processing
# merged_df.drop(['imdb_id','age_certification'], axis=1, inplace=True) # old way
merged_df = merged_df.drop(['imdb_id','age_certification'], axis=1)


# Replacing NULL values with N/A in Description column
# merged_df['description'].fillna('N/A', inplace = True) old way
merged_df['description'] = merged_df['description'].fillna('N/A')

# Replacing Null Categorical values with the mode of that column
for i in merged_df[['genre', 'production_country']]:
    merged_df[i] = merged_df[i].fillna(merged_df[i].mode()[0])

# Median Imputation

# Extracting non NULL records into a separate DataFrame to fit for MICE Imputation
non_null_df = merged_df[~merged_df.isnull().any(axis=1)]

# Use Median Imputation to fill in missing values
columns_to_impute = ['imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
imputer = SimpleImputer(strategy='median')
imputed_values = imputer.fit_transform(merged_df[columns_to_impute])

# Using MICE Imputation to impute Missing Values
merged_df.loc[:, columns_to_impute] = imputed_values

# Content based filtering
# Designing recommender system based on plot
# Adding streaming platform section in the df

# Generating a list to add the streaming platform to each title
platform = []
for i in merged_df['id']:
    movie_streaming = []
    if i in netflix_titles['id'].values:
        movie_streaming.append('Netflix')

    platform.append(movie_streaming)

# Setting the list value into the Streaming Platform column
merged_df['streaming_platform'] = platform

# Separating dat in TV shows and movies

# Extracting Movies from the Dataset consisting of both Movies and TV Shows
movies = merged_df[merged_df['type'] == 'MOVIE'].copy().reset_index()
movies = movies.drop(['index'], axis=1)

# Extracting TV Shows from the Dataset consisting of both Movies and TV Shows
shows = merged_df[merged_df['type'] == 'SHOW'].copy().reset_index()
shows = shows.drop(['index'], axis=1 )

# Term Frequency-Inverse Document Frequency (TF-IDF)

# Define a TF-IDF Vectorizer Object to remove English stop words
tfidf = TfidfVectorizer(stop_words = 'english')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix_movies = tfidf.fit_transform(movies['description'])
tfidf_matrix_shows = tfidf.fit_transform(shows['description'])

# Cosine similarity

# Compute the cosine similarity matrix
cosine_sim_movies = linear_kernel(tfidf_matrix_movies, tfidf_matrix_movies)
cosine_sim_shows = linear_kernel(tfidf_matrix_shows, tfidf_matrix_shows)

indices_movies = pd.Series(movies.index, index=movies['title'])
indices_shows = pd.Series(shows.index, index=shows['title'])

def get_title(title,indices):
    # Function to create the 'index searcher' that searches for user's title index
    try:
        index = indices[title]
    except:
        print("\n  Title not found")
        return None

    if isinstance(index, np.int64):
        return index

    else:
        rt = 0
        print("Select a title: ")
        for i in range(len(index)):
            # print(f"{i} - {movies['title'].iloc[index[i]]}", end=' ')
            print(f"{i} - {movies['title'].iloc[index.iloc[i]]}", end=' ')
            print(f"({movies['release_year'].iloc[index.iloc[i]]})")
        rt = int(input())
        return index[rt]

# for movies

import streamlit as st

def get_recommendations_movie(title, cosine_sim=cosine_sim_movies):
    # Function that takes a movie title as input and outputs top 10 similar movies in return

    title = get_title(title, indices_movies)
    if title == None:
        st.write("Movie not found.")
        return

    idx = indices_movies[title]

    # Display the selected movie and its release year
    st.write(f"**Title:** {movies['title'].iloc[idx]} | **Year:** {movies['release_year'].iloc[idx]}")
    st.write('---')

    # print(f"Title: {movies['title'].iloc[idx]} |  Year: {movies['release_year'].iloc[idx]}")

    # print('**' * 40)

    # Fetching the pairwise similarity scores of all movies for that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sorting the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Extracting the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Fetching the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # print(movies[['title', 'release_year','streaming_platform']].iloc[movie_indices])

    # print('**' * 40)
    st.write("**Top 10 Similar Movies:**")
    st.write(movies[['title', 'release_year', 'streaming_platform']].iloc[movie_indices])

    st.write('---')

# for shows

def get_recommendations_show(title, cosine_sim=cosine_sim_shows):
    # Function that takes a TV show title as input and outputs top 10 similar movies in return
    title = get_title(title, indices_shows)
    if title == None:
        st.write("TV show not found.")
        return

    idx = indices_shows[title]

    # print(f"Title: {shows['title'].iloc[idx]} | Year: {shows['release_year'].iloc[idx]}")

    # print('**' * 40)

    # Display the selected TV show and its release year
    st.write(f"**Title:** {shows['title'].iloc[idx]} | **Year:** {shows['release_year'].iloc[idx]}")
    st.write('---')

    # Fetching the pairwise similarity scores of all shows for that show
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sorting the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Extracting the scores of the 10 most similar shows
    sim_scores = sim_scores[1:11]

    # Fetching the show indices
    show_indices = [i[0] for i in sim_scores]

    # print(shows[['title', 'release_year', 'streaming_platform']].iloc[show_indices])

    # print('**' * 40)

    # Display the recommendations in the Streamlit app
    st.write("**Top 10 Similar TV Shows:**")
    st.write(shows[['title', 'release_year', 'streaming_platform']].iloc[show_indices])

    st.write('---')





