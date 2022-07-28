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
        movies = pd.read_csv(moviepath, error_bad_lines=False)
        credits = pd.read_csv(creditpath, error_bad_lines=False)
        movies.shape, credits.shape

        movies = movies.merge(credits,on='title')
        movies.shape


        movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

        movies.dropna(inplace=True)

## any values repeated

        movies.duplicated().sum()

## no repeated values

        movies.iloc[0]['genres']

## see it has 3 types of genres: Action, Adveture, Fantasy and Science Fiction

## genre is String of lists
## String of lists: lets convert it to Lists

        import ast

        ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

## see how it changed the thing

        def convert(text):
            arr = []
            for i in ast.literal_eval(text):
                arr.append(i['name'])
    
            return arr

        movies['genres'] = movies['genres'].apply(convert)


        movies['keywords'] = movies['keywords'].apply(convert)


        movies.iloc[0]['cast']

## has a huge dataset
## lets just take the top 3 actors and thats it

        def convert_cast(text):
            arr= []
            count = 0
            for i in ast.literal_eval(text):
                if count<3:
                    arr.append(i['name'])
                    count+=1
            return arr

        movies['cast'] = movies['cast'].apply(convert_cast)


        movies['crew'][0]

## we will just use the director tag from this

        def convert_crew(text):
            arr = []
            for i in ast.literal_eval(text):
                if i['job'] == 'Director':
                    arr.append(i['name'])
            return arr

        movies['crew'] = movies['crew'].apply(convert_crew)


        movies['overview'][0]

## its in the string format, lets convert it to list

        movies['overview'] = movies['overview'].apply(lambda x: x.split())
        movies['overview'][0]



## names have gaps between them, so will be treated as 2 diff entities
## so lets replace space with undeescore

        arr = ['cast','crew','genres','keywords']
        for j in arr:
            movies[j] = movies[j].apply(lambda x: [i.replace(" ","_") for i in x])

      

        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

        new_df = movies[['movie_id','title','tags','genres']]


## lets make the tags to string

        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


        new_df['tags'][0]

## its always recommended to put your text in lower case.
## so lets  convert to lower case

        new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

## we need to stem the whole tags thing
## since dance same as dancing but treated as 2 diff entities

        import nltk 
        from nltk.stem.porter import PorterStemmer
        ps = PorterStemmer()

        def stem(text):
            y = []
    
            for i in text.split():
                y.append(ps.stem(i))
        
            return (" ".join(y))

        new_df['tags'] = new_df['tags'].apply(stem)

        new_df['tags'][0]


        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=5000,stop_words='english')

        vector = cv.fit_transform(new_df['tags']).toarray()
        vector.shape

## 4806 movies
## 5000 words

        cv.get_feature_names()[:20]

## starting 20 most common words

        len(cv.get_feature_names())

## since we are dealing in the higher dimesnions
## euclidia distance will fail as it wont be that reliable in higher dimensions
### simply we called this as curse of dimensionality

## so for that we will be using the cosine distance
## that is angle between the two vectors


## less the distance: more the similarity means less angle between two vectors


        from sklearn.metrics.pairwise import cosine_similarity
        cosine_similarity(vector)

## here we are taking distace(cosine angle) of each movie with otheer movie

        cosine_similarity(vector).shape

## since 4806 movies: so 4806 distanced between one and rest of the movies

        similarity = cosine_similarity(vector)

## this matrix will have diagonal as 1
## since similarity will be 1 of each movie with itself

## index of any movie
        index = new_df[new_df['title']=='Batman Begins'].index[0]


        sorted(similarity[index],reverse=True)
## but index is not being shown

        sorted(list(enumerate(similarity[0])),reverse=True, key = lambda x:x[1])


        def recommend(movie):
            index = new_df[new_df['title']==movie].index[0]
            distance = sorted(list(enumerate(similarity[index])),reverse=True, key = lambda x:x[1])
            for i in distance[1:11]:
                print(new_df.iloc[i[0]].title)
        st.text(recommend(url_file))
