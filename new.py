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
# To install this module, run:
# python -m pip install Pillow
from io import BytesIO
from PIL import Image
from PIL import ImageDraw
import json
import pandas as pd

st.set_page_config(layout="wide")

st.sidebar.header('A website using Azure Api')
st.sidebar.markdown('Face Api,Translator Api')


app_mode = st.sidebar.radio(
    "",
    ("About Me","Face Recognization","Object Detection","Translator"),
)


st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('N.V.Suresh Krishna | sureshkrishnanv24@gmail.com https://github.com/sureshkrishna123')



if app_mode =='Object Detection':
    
    
    st.markdown("<h1 style='text-align: center; color: skyblue; '> Object Recognition </h1>", unsafe_allow_html=True)

    st.title("Object Recognition(Powered by Azure)")

    st.markdown("Using Azure I build to **_Object_ detection** , it identify and analyse the image.")
    st.text("Detect the objects in images")

    url_file =  title = st.text_input('Paste image address URL')
    button_translate=st.button('Click me',help='To give the image')

    if button_translate and url_file :
        bookpath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/books.csv'
        moviepath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/tmdb_5000_movies.csv'
        creditpath = 'https://raw.githubusercontent.com/noahjett/Movie-Goodreads-Analysis/master/tmdb_5000_credits.csv'
        movies_df = pd.read_csv(moviepath, error_bad_lines=False)
        credits_df = pd.read_csv(creditpath, error_bad_lines=False)
        credits_df.columns = ['id','title','cast','crew']
        movies_df = movies_df.merge(credits, on="id")
        
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

        cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 


        movies_df = movies_df.reset_index()
        indices = pd.Series(movies_df.index, index=movies_df['title'])
        
        indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()


        def get_recommendations(title, cosine_sim=cosine_sim):
          idx = indices[title]
          similarity_scores = list(enumerate(cosine_sim[idx]))
          similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
          similarity_scores= sim_scores[1:11]
    # (a, b) where a is id of movie, b is similarity_scores

          movies_indices = [ind[0] for ind in similarity_scores]
          movies = movies_df["title"].iloc[movies_indices]
          return movies


        
        st.text("Recommendations for The Dark Knight Rises")
        print(get_recommendations(url_file), cosine_sim2)


        



