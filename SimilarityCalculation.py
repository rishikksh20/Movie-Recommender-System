#  Movie Recommender System
#  Author: Rishikesh
#  Date: 13th Feb 2016  
#

import pandas as pd
import scipy
import sklearn

r_cols=['user_id','movie_id','rating']
ratings=pd.read_csv('ml-latest/ratings.csv',names=r_cols,usecols=range(3))
ratings.head()
m_cols=['movie_id','title']
movies=pd.read_csv('ml-latest/movies.csv',names=m_cols,usecols=range(2))
movies.head()
ratings=pd.merge(movies,ratings)
ratings.head()
movieRatings=ratings.pivot_table(index=['user_id'],columns=['title'],values='rating',aggfunc = lambda x: x)
movieRatings.head()
starWarsRating= movieRatings["Round Midnight (1986)"]
starWarsRating.head()