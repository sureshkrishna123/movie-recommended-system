import streamlit as st
import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
import numpy as np
# To install this module, run:
# python -m pip install Pillow
from io import BytesIO
from PIL import Image
from PIL import ImageDraw
import json
import pandas as pd

st.set_page_config(layout="wide")

st.sidebar.header('A Movie Recommended Sysem Web app')



app_mode = st.sidebar.radio(
    "",
    ("About Me","Movie Recommended System"),
)


st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('N.V.Suresh Krishna | sureshkrishnanv24@gmail.com https://github.com/sureshkrishna123')

if app_mode =='About Me':
    st.image('images/pic.jpg', use_column_width=True)
    st.markdown('''
              # About Me \n 
                Hey this is N.V.Suresh Krishna. \n
                
                
                Also check me out on Social Media
                - [git-Hub](https://github.com/sureshkrishna123)
                - [LinkedIn](https://www.linkedin.com/in/suresh-krishna-nv/)
                - [Instagram](https://www.instagram.com/worldofsuresh._/)
                - [Portfolio](https://sureshkrishna123.github.io/sureshportfolio/)\n
                ''')

if app_mode =='Movie Recommended System':
    
    
    st.markdown("<h1 style='text-align: center; color: skyblue; '> Movie Recommended System </h1>", unsafe_allow_html=True)
    st.image(os.path.join('./images','Screenshot (85).png'),use_column_width=True )
    st.title("Movie Recommended System")

    st.markdown("It will suggest you the relevant movie.")
    mode = st.radio(
    "",
    ("By Movie Name","By Genre"),
    )
    if mode=='By Movie Name':
        Movie_name = st.text_input('Type the movie name')
        button_translate=st.button('Click me',help='To suggest a relevant movie')
        if button_translate and Movie_name :
            bookpath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/books.csv'
            moviepath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/tmdb_5000_movies.csv'
            creditpath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/tmdb_5000_credits.csv'
            movies_df = pd.read_csv(moviepath, error_bad_lines=False)
            credits_df = pd.read_csv(creditpath, error_bad_lines=False)
            credits_df.columns = ['id','title','cast','crew']
            movies_df = movies_df.merge(credits_df, on="id")
            movies_df.rename(columns = {'original_title':'title'}, inplace = True)
        
            from ast import literal_eval
            features = ["cast", "crew", "keywords", "genres"]

            for feature in features:
              movies_df[feature] = movies_df[feature].apply(literal_eval)

            def get_director(x):
              for i in x:
                if i["job"] == "Director":
                    return i["name"]
              return np.nan

            def get_list(x):
              if isinstance(x, list):
                names = [i["name"] for i in x]

                if len(names) > 3:
                    names = names[:3]

                return names
              return []
            movies_df["director"] = movies_df["crew"].apply(get_director)
            features = ["cast", "keywords", "genres"]
            for feature in features:
                movies_df[feature] = movies_df[feature].apply(get_list)              
        
        
            def clean_data(row):
              if isinstance(row, list):
                return [str.lower(i.replace(" ", "")) for i in row]
              else:
                if isinstance(row, str):
                    return str.lower(row.replace(" ", ""))
                else:
                    return ""
            features = ['cast', 'keywords', 'director', 'genres']
            for feature in features:
                movies_df[feature] = movies_df[feature].apply(clean_data)
        
        
            def create_soup(features):
                return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])
            movies_df["soup"] = movies_df.apply(create_soup, axis=1)


            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            count_vectorizer = CountVectorizer(stop_words="english")
            count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

            cosine_sim = cosine_similarity(count_matrix, count_matrix) 


            movies_df = movies_df.reset_index()
            indices = pd.Series(movies_df.index, index=movies_df['title'])
        
            indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()


            def get_recommendations(title, cosine_sim=cosine_sim):
              idx = indices[title]
              similarity_scores = list(enumerate(cosine_sim[idx]))
              similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
              similarity_scores= similarity_scores[1:11]
    # (a, b) where a is id of movie, b is similarity_scores

              movies_indices = [ind[0] for ind in similarity_scores]
              movies = movies_df["title"].iloc[movies_indices]
              return movies


        
            st.text("Recommendations for "+Movie_name)
            st.text(get_recommendations(Movie_name))
            st.text("##########################")
            s.text("Enjoy the movie :)")

    if mode=='By Genre':

        detect_select=st.selectbox("select Genre from the list",['Action','Adventure','Comedy','Fantasy','Science Fiction',' Crime','Thriller','Horror','Romance','TV Movie','Drama','Animation','Family','Western'],key=1)
        button_detect=st.button('Click me',help='To detect language')                         
        if button_detect and detect_select:
            bookpath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/books.csv'
            moviepath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/tmdb_5000_movies.csv'
            creditpath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/tmdb_5000_credits.csv'
            movies_df = pd.read_csv(moviepath, error_bad_lines=False)
            credits_df = pd.read_csv(creditpath, error_bad_lines=False)
            credits_df.columns = ['id','title','cast','crew']
            movies_df = movies_df.merge(credits_df, on="id")
            movies_df.rename(columns = {'original_title':'title'}, inplace = True)
        
            for genre in movies_df['genres']:
                if detect_select in genre:
                    indices =movies_df[movies_df['genres'] == genre].index.values
                    movies = movies_df["title"].iloc[indices]
            st.text(movies)
            st.text("##########################")
            s.text("Enjoy the movie :)")



