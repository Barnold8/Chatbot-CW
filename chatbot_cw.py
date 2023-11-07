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
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

### PLAYLIST MANAGEMENT CHATBOT

# NLTK DOWNLOADS

nltk.download('vader_lexicon')

# END OF NLTK DOWNLOADS

# GLOBAL VARIABLE DECLARATION
n_param = 3
debug = False
running = True
user_name = "Unknown user"
lemmatizer = WordNetLemmatizer()

already_asked = False
# END OF GLOBAL VARIABLE DECLARATION

def intent(user_input: str):
    """
    Desc:
        This functions' is fairly complex for its use. I use the Naive Bayes classifier to help
        predict what the users intent is via their speech. In this we, have a dictionary of 
        intents, these will help us relate to what the user wants via their speech. 
        We input these intents into an X and Y to help the classfier learn what everything is.
        The code will the turn it into a BOW model to help pre-process the text for classification.
        With this we can pass it to the classifier. We can use the predict function which will use some
        underlying formulae to predict the closest likelyhood intent. In my own opinion, I think it should
        use cosine similarity to find the closest vector relative but it may not use that formula.

    return: string that says what the intent was

    """
    # May be a better idea to use a vector space model and then cosine similarity

    # The intents to understand what the user is saying 
    intents = [ 

        {"intent": "greeting", "examples": ["hi there", "hello", "hey"]},
        {"intent": "thanks", "examples": ["thank you", "thanks a lot"]},
        {"intent": "name_retrieval","examples":["my name is","please call me","I want to be known as","I am","hello there, I am","hey, I am"]},
        {"intent": "transaction","examples": ["playlist","i want to edit my playlist","whats on my playlist"]},
        {"intent": "stop", "examples" : ["stop the application","stop listening","stop","Goodbye", "Bye", "See you later"]},
        {"intent": "void","examples": ["nothing","nevermind", "im unsure", "i don't know"]},
        {"intent": "help","examples": ["help","help me please","i require help","i need help"]},
        {"intent": "yes_no","examples":["i agree","yes please","yes","no","no thank you","i do not agree"]}

    ]

    X = [] # The X, input data / features
    y = [] # The Y, what we are learning

    # add examples and intents accordingly to X and Y training
    for intent_data in intents: 
        intent = intent_data["intent"]
        examples = intent_data["examples"]
        X.extend(examples)
        y.extend([intent] * len(examples))

    # vectorize and classify our intents
    vect_and_class = Pipeline([ 
        ('vectorizer', CountVectorizer()),  # Convert text data to a bag-of-words representation
        ('classifier', MultinomialNB())     # Naive Bayes classifier
    ])

    # fit the data in our pipeline
    vect_and_class.fit(X, y)

    # store the users intent according to the input they gave
    predicted_intent = vect_and_class.predict([user_input])

    # return the most likely / most confident, intent
    return predicted_intent[0]

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
    """
    Desc:
        This is just going to be the basis for syntatic aggregation if i need to expand upon it

    return: aggregated string via syntax
    """
    subj1, verb1, obj1 = s1.split()
    subj2, verb2, obj2 = s2.split()

    if subj1 == subj2 and verb1 == verb2:
        return f"{ subj1 } { verb1 } { obj1 } and { obj2 }."
    else:
        return None

def sentiment(inp: str, low_bound = 0, high_bound = 0) -> int:

    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_attribute = sentiment_analyzer.polarity_scores(inp)
    compound = sentiment_attribute["compound"]
   
    # print(f"Sentence analysis: \n\tSentence: {inp}\n\tAnalysis: {sentiment_attribute}")

    # print(sentiment_attribute)

    if compound > high_bound:
        return 1
    elif compound < low_bound:
        return -1
    else:
        return 0

def POS(user_input: str, grab_by_tag = None): # Possible fix to singular name being seen as RB, train POS on set of names being NNP
    """
    Desc:
        This function works by tokenizing an incoming string and then tagging each word using the pre-trained model
        provided by NLTK. This function allows the developer to pick a tag to find within a string, so say they wanted NNP from
        a string, they would then get a list of tuples where each tuple is of type NNP. The main purpose for POS thus far is 
        to grab a name from a sentence. 
    
    return: N/A
    
    """

    # ## TEXT PRE PROCESSING STAGE
    # custom_punctuation  = string.punctuation + "’" + "-" + "‘" + "-"  # Have a string containing punctuation we want to remove from our text
    # # custom_punctuation.replace(".", "")                             # Remove the full stop in the custom punctuation (We need this to identify what and what isnt a sentence)

    # user_input = "".join([char for char in user_input if char not in custom_punctuation])
    # ##

    # tokenized_txt = word_tokenize(user_input.lower())

    user_input = string_preprocess(user_input)

    tokenized_txt = word_tokenize(user_input)



    # Data sourced from https://github.com/dominictarr/random-name/blob/master/first-names.txt
    with open("Data/names.txt") as file:

        names = file.readlines()
        names = [name.lower() for name in names]
        names = [name.strip() for name in names]

        for i in range(len(tokenized_txt)):
            if tokenized_txt[i] in names:
                tokenized_txt[i] = tokenized_txt[i].capitalize()

    tags = nltk.pos_tag(tokenized_txt)

    if grab_by_tag:
        group = []
        for tag in tags:
            if tag[1] == grab_by_tag:
                group.append(tag)
        return group
    else:
        return tags

def getName(inp: str, attempts: int) -> None:
    
    ALLOWED_ATTEMPS = 3 # this is a constant and must not be accessed

    intended_result = intent(inp)
    print(intended_result)

    if intended_result != "name_retrieval":
        intent_decider(intended_result, inp)
        return

    if attempts >= ALLOWED_ATTEMPS:
        print("I am sorry, I really struggled to catch your name, I do apologise.")
        user_name = "Unknown User" # keep consistency for user name
    
    if len(inp.split(" ")) < 2:
        # A very bad bodge. But it works (singular words dont work well for identifying NNP for tags)
        inp = "please call me " + inp

    try:
    
        user_name = POS(inp,"NNP")[0][0]
        user_input = input(f"So, you would like me to refer to you as {user_name}?\n")
        intended_result = intent(user_input)

        if intended_result != "yes_no": # Continue this to intercept the new intent of a user here 
            print(f"WHATTTTT: {intended_result}")
        
        sentiment_input = sentiment(user_input)

        if sentiment_input <= 0:

            print(sentiment_input)
            attempts += 1
            getName(input(f"What would you like me to call you?\n"),attempts)

        else: 

            print(f"Nice to meet you {user_name}")
            return 
        
    except IndexError as name_error:
        print("Sorry, I couldn't quite catch your name. I am only limited to english names.")
        attempts += 1
        getName(input(f"What would you like me to call you?\n"),attempts)



def intent_decider(intent: string, inp: string) -> None:
    """
        Desc:
            The purpose of this function is just to clean up code for the 
            god awful set of if statements that help the code logically 
            do stuff after an intent is found
        return: None
    """
    if intent == "name_retrieval":
        user_name = getName(inp,0)


def string_preprocess(inp: string) -> str:
    """
        Desc:
            This function just does what it says on the tin. It is the preprocessing
            step. In this we remove all punctuation from the strings to allow for 
            consistency in our data 
        return: string
    """
    custom_punctuation  = string.punctuation + "’" + "-" + "‘" + "-" 
    inp = "".join([char for char in inp if char not in custom_punctuation])
    return inp.lower()

# Program loop

# print("\nHi, i'm JAMSIE (Just Awesome Music Selection and Interactive Experience.), how can I help today? P.S, if you ask for help, ill provide a list of my functionality! :D")

while running :

    if already_asked:
        prompt = input("\nWhat else can I help you with?\n")
    else:
        already_asked = True
        prompt = input("\nWhat can I help you with?\n")


    user_intent = intent(prompt)

    if user_intent == "stop":
        print(f"Goodbye {user_name}!")
        break
    else:
        intent_decider(user_intent,prompt)




## HELP DOCUMENTATION

# https://www.nltk.org/book/ch05.html

# https://www.nltk.org/api/nltk.sentiment.vader.html

# https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python#:~:text=Cosine%20Similarity%20computes%20the%20similarity,the%20cosine%20similarity%20is%201