import nltk , re , pprint , string
import numpy as np
from nltk import word_tokenize , sent_tokenize
from nltk . util import pad_sequence
from nltk . lm import MLE , Laplace
from nltk . lm . preprocessing import pad_both_ends , padded_everygram_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# GLOBAL VARIABLE DECLARATION
n_param = 3
debug = False
running = True



# END OF GLOBAL VARIABLE DECLARATION


def intent(user_input: str):

    # The intents to understand what the user is saying 
    intents = [ 

    {"intent": "greeting", "examples": ["Hi there!", "Hello!", "Hey"]},
    {"intent": "goodbye", "examples": ["Goodbye", "Bye", "See you later"]},
    {"intent": "weather", "examples": ["What's the weather today?", "Tell me the weather forecast"]},
    {"intent": "thanks", "examples": ["Thank you!", "Thanks a lot"]},
    {"intent": "stop", "examples" : ["stop the application","stop listening","stop"]},
    ]

    X = [] # The X, input data/ features
    y = [] # The Y, what we are learning


    for intent_data in intents:
        intent = intent_data["intent"]
        examples = intent_data["examples"]
        X.extend(examples)
        y.extend([intent] * len(examples))


    text_clf = Pipeline([
        ('vectorizer', CountVectorizer()),  # Convert text data to a bag-of-words representation
        ('classifier', MultinomialNB())  # Multinomial Naive Bayes classifier
    ])

    text_clf.fit(X, y)

    predicted_intent = text_clf.predict([user_input])
    print(f"Predicted intent: {predicted_intent[0]}")


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
    

# Program loop
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

while running :
    user_input = input("Say something: ").lower()

    if "stop" in user_input:
        running = False
    else :
        print (f'You are searching for { user_input}')
    intent(user_input)

# Page 15 
