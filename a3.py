# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    movies_t = []
    for i in movies['genres']:
        movies_t.append(tokenize_string(i))
    #print(movies_t)
    movies['tokens'] = movies_t
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    vocab_set = set()
    vocab = {}
    #print(movies)
    movie_id_to_token_freq ={}
    term_freq ={}
    term_freq_list =[]
    for index,rows in movies.iterrows():
        for i in (rows['tokens']):
            vocab_set.add(i)
            if i in term_freq:
                term_freq[i] +=1
            else:
                term_freq[i] =1
        movie_id_to_token_freq[rows['movieId']] = term_freq
        term_freq_list.append(term_freq)
        term_freq ={}
        
    #print(vocab_set)
    i=0
    for feature in sorted(vocab_set):
        vocab[feature] = i
        i+=1
    i=0
    #print(vocab)
    
    #Used to calculate tfidf
    uniq_f_to_doc ={}
    num_doc = len(movies)
    #print(num_doc)
    for index,rows in movies.iterrows():
        for i in set(rows['tokens']):
            if i in uniq_f_to_doc:
                uniq_f_to_doc[i] +=1
            else:
                uniq_f_to_doc[i] =1
    #print(uniq_f_to_doc)
    #print(movie_id_to_token_freq)        
    
    
    
    #Used for csr matrix
    csr_mat =[]
    data_mat =[]
    row =[]
    column =[]
    #data =[]
    j=0
    for index,r in movies.iterrows():
        for i in set(r['tokens']):
            row.append(j)
            column.append(vocab[i])
            #print(rows['movieId'])
            if r['movieId'] in movie_id_to_token_freq:
                tokens_data =(movie_id_to_token_freq[r['movieId']])
                val =tokens_data[i]
                #print(movie_id_to_token_freq[rows['movieId'][i])
                max_val=(max(movie_id_to_token_freq[r['movieId']].values()))
                #print(num_doc)
                #print(uniq_f_to_doc[i])
                tf_idf_val = (val / max_val) * math.log10(num_doc/uniq_f_to_doc[i])
                data_mat.append(tf_idf_val)
        #print(len(row))
        #print(len(column))
        #print(len(data_mat))
        #print(row)
        #print(column)
        #print(data_mat)
        csr_mat.append(csr_matrix((data_mat, (row, column)), shape=(1, len(vocab))))
        data_mat =[]
        row =[]
        column =[]
        
    
    movies['features'] = pd.Series(csr_mat, index=movies.index)
    #print(movies)
    return tuple((movies,vocab))
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    cos_sim_val = (a.toarray()).dot(((b.toarray())).transpose()) / (np.sqrt(a.multiply(a).sum(axis=1))).dot(
        (np.sqrt(b.multiply(b).sum(axis=1))).transpose())
    
    return ((cos_sim_val.item(0)))
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    predicted = []
    
    for i in range(len(ratings_test)):
        #print(ratings_test['movieId'].iloc[i])
        mov_id_test = ratings_test['movieId'].iloc[i]
        user_test = ratings_test['userId'].iloc[i]
        mov_id_train = ratings_train[ratings_train.userId == user_test]
        ratings = ratings_train[ratings_train.userId == user_test]
        #print(mov_id_test)
        #print(user_test)
        #print(mov_id_train)
        #print(ratings)
        avg_sum = []
        cosine_sum = []
        
        for index, row in mov_id_train.iterrows():
            X1 = movies[movies.movieId == mov_id_test].features.iloc[0]
            X2 = movies[movies.movieId == row.movieId].features.iloc[0]
               
            cosine = cosine_sim(X1, X2)
            #print(cosine)    
            if(cosine > 0):
                avg_sum.append(row.rating * cosine)
                cosine_sum.append(cosine)
            
        
        if(len(avg_sum) <= 0):
            average = np.mean(ratings.rating)
        else:
            average = np.sum(avg_sum)/np.sum(cosine_sum)
        
        predicted.append(average)
    return np.array(predicted)
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
