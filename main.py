from tfidf import TF_IDF as tfidf
from corpy import Corpy as cp
import pandas as pd

# short corpus for the example
corpus = ["gengar is my favorite pokemon",
          "i like the ghost type pokemon",
          "mega gengar is the best mega evolution in pokemon"]

# need a corpy object for the correct use for the corpus 
corpy = cp(corpus)

tf_idf = tfidf(corpy)

# use pandas for a better view of the results
print("--- TF ---")
df = pd.DataFrame(tf_idf.TF, columns=corpy.get_vocabulary())
print(df)
print("--- IDF ---")
df2 = pd.DataFrame(tf_idf.IDF, columns=corpy.get_vocabulary())
print(df2)
print("--- TF-IDF ---")
df3 = pd.DataFrame(tf_idf.TF_IDF, columns=corpy.get_vocabulary())
print(df3)