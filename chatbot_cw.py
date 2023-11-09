import nltk , re , pprint , string
import numpy as np
from random import randint
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv

### PLAYLIST MANAGEMENT CHATBOT

def cls() -> None:
    """
        Desc:
            Clears console, multi platform using os.name
        return: None
    """
    os.system('cls' if os.name=='nt' else 'clear')

def parseCSV(file: str) -> list[tuple]:

    with open("Data/COMP3074-CW1-Dataset.csv", "r", encoding="UTF-8") as file:

        reader = csv.DictReader(file)
        
        qa = [] # questions and answers

        for elem in reader:
            qa.append((elem["Question"],elem["Answer"]))
        
        return qa


# NLTK DOWNLOADS
print("Just downloading some needed data. Please wait...")
nltk.download('vader_lexicon')
os.system("cls")    # Clear downloads output

# END OF NLTK DOWNLOADS

# GLOBAL VARIABLE DECLARATION
n_param = 3
debug = False
running = True
user_name = None
lemmatizer = WordNetLemmatizer()
hi_string = "\nHi, i'm JAMSIE (Just Awesome Music Selection and Interactive Experience.), how can I help today? P.S, if you ask for help, ill provide a list of my functionality! :D"
already_asked = False
qa_data = parseCSV("COMP3074-CW1-Dataset.csv") # this is use for the general question answering 
# END OF GLOBAL VARIABLE DECLARATION

def intent_help() -> None:
    """
        Desc:
            This is just a wrapper function for printing help
        return: None
    """
    
    print("I can sure help! Here's a list of what I can do:\nNothing :( ")

def intent(user_input: str) -> str:
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

        {"intent": "greeting", "examples": ["hi there", "hello", "hey","hello there","hi","howdy","greetings","salutations","good day","hey there"]},
        {"intent": "thanks", "examples": ["thank you","thanks","i appreciate it","im grateful","im thankful","much obliged","i owe you one","that means a lot","im so grateful",]},
        {"intent": "name_retrieval","examples":["my name is","please call me","I want to be known as","I am","hello there, I am","hey, I am"]},
        {"intent": "transaction","examples": ["playlist","i want to edit my playlist","whats on my playlist"]},
        {"intent": "stop", "examples" : ["stop the application","stop listening","stop","Goodbye", "Bye", "See you later"]},
        {"intent": "void","examples": ['lets change the topic', 'nevermind', 'im neutral on the subject', 'i cant say for sure','im uncertain', 'i dont know', 'im not sure', 'i have nothing to add', 'i have no idea', 'its unclear to me']},
        {"intent": "help","examples": ["help","help me please","i require help","i need help"]},
        {"intent": "yes_no","examples":["i agree","yes please","yes","no","no thank you","i do not agree"]},
        {"intent": "question","examples": ['i need some information', 'im curious about something', 'id like to inquire', 'im wondering about something', 'i have a query', 'im seeking clarification', 'i want to learn more', 'im interested in knowing', 'im looking for answers']}

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
    global user_name
    
    ALLOWED_ATTEMPS = 3 # this is a constant and must not be accessed

    intended_result = intent(inp)

    if intended_result != "name_retrieval":
        intent_decider(intended_result, inp)
        return user_name

    if attempts >= ALLOWED_ATTEMPS:
        print("I am sorry, I really struggled to catch your name, I do apologise.")
        return user_name 
    
    if len(inp.split(" ")) < 2:
        # A very bad bodge. But it works (singular words dont work well for identifying NNP for tags)
        inp = "please call me " + inp

    try:
    
        user_name_input = POS(inp,"NNP")[0][0]
        user_input = input(f"So, you would like me to refer to you as {user_name_input}?\n")
        intended_result = intent(user_input)

        if intended_result != "yes_no": # Continue this to intercept the new intent of a user here 
            intent_decider(intended_result, inp)
            return user_name
        
        sentiment_input = sentiment(user_input)

        if sentiment_input <= 0:

            print(sentiment_input)
            attempts += 1
            getName(input(f"What would you like me to call you?\n"),attempts)

        else: 
            print(f"Nice to meet you {user_name_input}")
            return user_name_input
        
    except IndexError as name_error:
        print("Sorry, I couldn't quite catch your name. I am only limited to english names.")
        attempts += 1
        getName(input(f"What would you like me to call you?\n"),attempts)

def intent_decider(intent: string, inp: string) -> None:
    global user_name
    """
        Desc:
            The purpose of this function is just to clean up code for the 
            god awful set of if statements that help the code logically 
            do stuff after an intent is found
        return: None
    """


    # {"intent": "greeting", "examples": ["hi there", "hello", "hey"]},
    # {"intent": "thanks", "examples": ["thank you", "thanks a lot"]},
    # {"intent": "name_retrieval","examples":["my name is","please call me","I want to be known as","I am","hello there, I am","hey, I am"]},
    # {"intent": "transaction","examples": ["playlist","i want to edit my playlist","whats on my playlist"]},
    # {"intent": "stop", "examples" : ["stop the application","stop listening","stop","Goodbye", "Bye", "See you later"]},
    # {"intent": "void","examples": ["nothing","nevermind", "im unsure", "i don't know"]},
    # {"intent": "help","examples": ["help","help me please","i require help","i need help"]},
    # {"intent": "yes_no","examples":["i agree","yes please","yes","no","no thank you","i do not agree"]}

    # print(f"Here is the intent: {intent}")

    if intent == "stop":
        if user_name:
            print(f"JAMSIE: Goodbye {user_name}!")
        else:
            print("JAMSIE: Goodbye!")
        exit(0)

    else:
        # why cant python have syntatically good switch case...
        if intent == "name_retrieval":
            user_name = getName(inp,0)
        elif intent == "greeting":
            greetings = ["Hello", "Hi", "Hey", "Howdy", "Greetings", "Salutations","Good day","Hey there"]
            print(f"JAMSIE: {greetings[randint(0,len(greetings))]}")
        elif intent == "thanks":
            print("JAMSIE: Not a problem, glad to help! :D")
        elif intent == "transaction":
            print("transaction")
        elif intent == "void":
            print("void")
        elif intent == "help":
            intent_help()
        elif intent == "question":
            if user_name != None:
                question = input(f"JAMSIE: What would you like to ask me {user_name}?")
            else:
                question = input(f"JAMSIE: What would you like to ask me?")
            print(f"Answer: {questionAnswer(qa_data,question)}")
        else:
            print("JAMSIE: I am unsure what you are asking of me, sorry. :(")

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

def similirityMatching(data: list[str], inp: str):

    vectorizer = TfidfVectorizer()

    vectorized_strings = vectorizer.fit_transform(data)

    vectorized_new_string = vectorizer.transform([inp])

    # Calculate cosine similarity between the new string and all vectors in the array
    cosine_similarities = cosine_similarity(vectorized_new_string, vectorized_strings)

    # if sum(cosine_similarities) == 0: # Input is dissimilar to everything given
    #     return None

    # Print the cosine similarities

    return np.argmax(cosine_similarities)

def questionAnswer(qa_package: list[tuple],question: string):

    questions = [x[0] for x in qa_package]

    return qa_package[similirityMatching(questions,"Who is john wayne")][1]

# Program loop

print(hi_string)
print("-"*len(hi_string))

while running :

    if already_asked:
        prompt = input("\nWhat else can I help you with?\n")
    else:
        already_asked = True
        prompt = input("\nWhat can I help you with?\n")

    user_intent = intent(prompt)

    intent_decider(user_intent,prompt)




# simalirityMatching()


## HELP DOCUMENTATION

# https://www.nltk.org/book/ch05.html

# https://www.nltk.org/api/nltk.sentiment.vader.html

# https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python#:~:text=Cosine%20Similarity%20computes%20the%20similarity,the%20cosine%20similarity%20is%201