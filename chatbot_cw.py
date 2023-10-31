import nltk , re , pprint , string

from nltk import word_tokenize , sent_tokenize
from nltk . util import pad_sequence
from nltk . lm import MLE , Laplace
from collections import Counter
from nltk . lm . preprocessing import pad_both_ends , padded_everygram_pipeline

# GLOBAL VARIABLE DECLARATION
n_param = 2
debug = False

# END OF GLOBAL VARIABLE DECLARATION

def tokenize(inp : str)-> list[str]:
    """
    Desc:
         takes input of type string, a human readable sentence and tokenizes it
    return: list of tokens of type string
    """
    custom_punctuation  = string.punctuation + "’" + "-" + "‘" + "-"  # Have a string containing punctuation we want to remove from our text
    custom_punctuation.replace(".", "")                               # Remove the full stop in the custom punctuation (We need this to identify what and what isnt a sentence)

    inp = inp.replace("\n"," ")

    "".join([char for char in inp if char not in custom_punctuation])  # Make the file contents into a singular string

    if debug:
        print("DEBUG: Processing inp string...")

    return [word_tokenize(sent) for sent in sent_tokenize(inp)]

def padder(inp : list[str]):
    """"""
    corpus,vocab = padded_everygram_pipeline(n_param,inp)
    lm = Laplace(n_param)
    lm.fit(corpus,vocab)
    return lm


print(tokenize("This is a form of a document"))