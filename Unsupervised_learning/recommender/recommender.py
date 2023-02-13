import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import pickle

MOVIES_LIST = pd.read_csv("movies_list.csv")

filename = "my_nmf_model.sav"
model = pickle.load(open(filename, "rb"))

def recommend_nmf(query, movies_list, model = "my_nmf_model.sav", k = 10):    
   
    """
    Filters and recommends the top k movies 
    for any given input query based 
    on a trained NMF model.

    Parameters
    ----------
    query : dict
        A dictionary of movies already seen. 
        Takes the form {"movie_A": 3, "movie_B": 3} etc
        
    model : pickle
        pickle model read from disk
        
    k : int, optional
        no. of top movies to recommend, by default 10
        
    """
    
    user = pd.DataFrame(list(query.items()), columns = ["title", "rating"])
    user.set_index("title", inplace = True)
    
    
    user_merge = pd.merge(movies_list, user, on = "title", how = "left")
   
    # calculate the score with the NMF model
    user_query = user_merge["rating"]
    user_query = user_query.fillna(0)
    user_query = np.array(user_query)
    
    user2 = np.array(user_query)
    user2 = user2.reshape(1, -1)
    user2 = model.transform(user2)
    
    
    Q = model.components_
    rec = np.dot(user2, Q)
    rec = pd.Series(rec[0], index = df_pivot.columns)
    
    # set zero score to movies allready seen by the user
    rec = pd.DataFrame(rec)
    
    for i in user.index:
        rec.drop(index = [i], inplace = True)
        
        
    # return the top-k highst rated movie ids or titles
    
    recommendations = rec.sort_values(by = 0, ascending = False).head(k)
    return recommendations.index.tolist()



if __name__ == '__main__':
    print(f"Your recommendations are:\n{', '.join(recommend_nmf(user_query, MOVIES_LIST, model, k = 10))}")
