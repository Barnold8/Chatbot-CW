import nltk , re , pprint , string
import numpy as np
from nltk import word_tokenize , sent_tokenize
from nltk . util import pad_sequence
from nltk . lm import MLE , Laplace
from collections import Counter
from nltk . lm . preprocessing import pad_both_ends , padded_everygram_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# GLOBAL VARIABLE DECLARATION
n_param = 3
debug = False
# END OF GLOBAL VARIABLE DECLARATION

class Intent:

    def __init__(self) -> None:
        pass


def tokenize(inp : str)-> list[str]:
    """
    Desc:
         takes input of type string, a human readable sentence and tokenizes it
    return: list of tokens of type string
    """
    inp = inp.lower()
    
    custom_punctuation  = string.punctuation + "'" + "-" + "‘" + "-"  # Have a string containing punctuation we want to remove from our text
    custom_punctuation.replace(".", "")                               # Remove the full stop in the custom punctuation (We need this to identify what and what isnt a sentence)

    inp = inp.replace("\n"," ")

    inp = "".join([char for char in inp if char not in custom_punctuation])  # Make the file contents into a singular string

    return [word_tokenize(sent) for sent in sent_tokenize(inp)]

def padder(inp : list[str]) -> nltk.lm.models.Laplace :
    """
    Desc:
        This takes in a list of tokens and then applies them to the Laplace statistical model. This helps us 
        make a ngram (using the global n_param variable)
    return: Laplace model
    """
    corpus,vocab = padded_everygram_pipeline(n_param,inp)
    lm = Laplace(n_param)
    lm.fit(corpus,vocab)

    return lm

def syntatic_aggregation(s1,s2):

    subj1, verb1, obj1 = s1.split()
    subj2, verb2, obj2 = s2.split()

    if subj1 == subj2 and verb1 == verb2:
        return f"{ subj1 } { verb1 } { obj1 } and { obj2 }."
    else:
        return None
    





# Page 15 
