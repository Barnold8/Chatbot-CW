# TODO: Transactions

import os
## ENSURE USER HAS NEEDED LIBS BY TESTING IMPORTS

try:
    import nltk , re , pprint , string
except ImportError:
    os.system("pip install nltk")
try:
    import numpy as np
except ImportError:
    os.system("pip install numpy")
try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    os.system("pip install scikit-learn")

## END OF IMPORT ENSURANCE

from random import randint
from nltk import word_tokenize , sent_tokenize
from nltk . util import pad_sequence
from nltk . lm import MLE , Laplace
from nltk . lm . preprocessing import pad_both_ends , padded_everygram_pipeline
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir
from os.path import isfile, join

import csv
import json 
from datetime import datetime


class PlaylistManager:

    def getSongs():
        files = [file for file in listdir("Data/Playlist") if isfile(join("Data/Playlist", file))]
        return [file for file in files if ".mp3" in file]

    def sortSongs():
        songs = PlaylistManager.getSongs()
        os.mkdir("Data/temp")


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

def loadJSON(file: str) -> dict:
    general_intents = None
    try:
        with open("Data/intents.json","r") as file:
            try:
                general_intents = json.load(file)
            except json.decoder.JSONDecodeError as JSONERR:
                print(f"FATAL ERROR: Cannot read json file. Error information\n\t{JSONERR}")
                exit(1)
    except FileNotFoundError as error:
        print(f"FATAL ERROR: Cannot load file. Error information\n\t{error}")
        exit(1)
    return general_intents

# NLTK DOWNLOADS
print("Just downloading some needed data. Please wait...")
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
os.system("cls")    # Clear downloads output

# END OF NLTK DOWNLOADS

# GLOBAL VARIABLE DECLARATION
n_param = 3
running = True
user_name = None
lemmatizer = WordNetLemmatizer()
hi_string = "\nHi, i'm JAMSIE (Just Awesome Music Selection and Interactive Experience.), how can I help today? P.S, if you ask for help, ill provide a list of my functionality! :D"
already_asked = False
qa_data = parseCSV("COMP3074-CW1-Dataset.csv") # this is use for the general question answering, load once for quicker processing
intents = loadJSON("Data/intents.json") # this is a nice concise place to keep intents rather than hardcoding them in to the chatbot
user_hobbies = []

# Got data from https://github.com/kootenpv/contractions/blob/master/contractions/data/contractions_dict.json
contractions = {
        "im": "i am",
        "ima": "i am about to",
        "imo": "i am going to",
        "ive": "i have",
        "ill": "i will",
        "illve": "i will have",
        "id": "i would",
        "idve": "i would have",
        "whatcha": "what are you",
        "amnt": "am not",
        "aint": "are not",
        "arent": "are not",
        "cause": "because",
        "cant": "cannot",
        "cantve": "cannot have",
        "couldve": "could have",
        "couldnt": "could not",
        "couldntve": "could not have",
        "darent": "dare not",
        "daresnt": "dare not",
        "dasnt": "dare not",
        "didnt": "did not",
        "dont": "do not",
        "doesnt": "does not",
        "eer": "ever",
        "everyones": "everyone is",
        "finna": "fixing to",
        "gimme": "give me",
        "gont": "go not",
        "gonna": "going to",
        "gotta": "got to",
        "hadnt": "had not",
        "hadntve": "had not have",
        "hasnt": "has not",
        "havent": "have not",
        "heve": "he have",
        "hes": "he is",
        "hell": "he will",
        "hellve": "he will have",
        "hed": "he would",
        "hedve": "he would have",
        "heres": "here is",
        "howre": "how are",
        "howd": "how did",
        "howdy": "how do you",
        "hows": "how is",
        "howll": "how will",
        "isnt": "is not",
        "its": "it is",
        "tis": "it is",
        "twas": "it was",
        "itll": "it will",
        "itllve": "it will have",
        "itd": "it would",
        "itdve": "it would have",
        "kinda": "kind of",
        "lets": "let us",
        "luv": "love",
        "maam": "madam",
        "mayve": "may have",
        "maynt": "may not",
        "mightve": "might have",
        "mightnt": "might not",
        "mightntve": "might not have",
        "mustve": "must have",
        "mustnt": "must not",
        "mustntve": "must not have",
        "neednt": "need not",
        "needntve": "need not have",
        "neer": "never",
        "o": "of",
        "oclock": "of the clock",
        "ol": "old",
        "oughtnt": "ought not",
        "oughtntve": "ought not have",
        "oer": "over",
        "shant": "shall not",
        "shant": "shall not",
        "shallnt": "shall not",
        "shantve": "shall not have",
        "shes": "she is",
        "shell": "she will",
        "shed": "she would",
        "shedve": "she would have",
        "shouldve": "should have",
        "shouldnt": "should not",
        "shouldntve": "should not have",
        "sove": "so have",
        "sos": "so is",
        "somebodys": "somebody is",
        "someones": "someone is",
        "somethings": "something is",
        "sux": "sucks",
        "thatre": "that are",
        "thats": "that is",
        "thatll": "that will",
        "thatd": "that would",
        "thatdve": "that would have",
        "em": "them",
        "therere": "there are",
        "theres": "there is",
        "therell": "there will",
        "thered": "there would",
        "theredve": "there would have",
        "thesere": "these are",
        "theyre": "they are",
        "theyve": "they have",
        "theyll": "they will",
        "theyllve": "they will have",
        "theyd": "they would",
        "theydve": "they would have",
        "thiss": "this is",
        "thisll": "this will",
        "thisd": "this would",
        "thosere": "those are",
        "tove": "to have",
        "wanna": "want to",
        "wasnt": "was not",
        "were": "we are",
        "weve": "we have",
        "well": "we will",
        "wellve": "we will have",
        "wed": "we would",
        "wedve": "we would have",
        "werent": "were not",
        "whatre": "what are",
        "whatd": "what did",
        "whatve": "what have",
        "whats": "what is",
        "whatll": "what will",
        "whatllve": "what will have",
        "whenve": "when have",
        "whens": "when is",
        "wherere": "where are",
        "whered": "where did",
        "whereve": "where have",
        "wheres": "where is",
        "whichs": "which is",
        "whore": "who are",
        "whove": "who have",
        "whos": "who is",
        "wholl": "who will",
        "whollve": "who will have",
        "whod": "who would",
        "whodve": "who would have",
        "whyre": "why are",
        "whyd": "why did",
        "whyve": "why have",
        "whys": "why is",
        "willve": "will have",
        "wont": "will not",
        "wontve": "will not have",
        "wouldve": "would have",
        "wouldnt": "would not",
        "wouldntve": "would not have",
        "yall": "you all",
        "yallre": "you all are",
        "yallve": "you all have",
        "yalld": "you all would",
        "yalldve": "you all would have",
        "youre": "you are",
        "youve": "you have",
        "youllve": "you shall have",
        "youll": "you will",
        "youd": "you would",
        "youdve": "you would have",
        "goodbye": "good bye"
}

# END OF GLOBAL VARIABLE DECLARATION

def lemmatizeString(inp: str) -> str:
    """
        Desc:
            This is just a wrapper function for some sort of unreadable code.
            It will tokenize the string, and lemmatize each word in said string 
            and then turn that back into a string. Not all strings
            should be lemmatized

        return: lemmatized string
    """
    return "".join([lemmatizer.lemmatize(word) + " " for word in word_tokenize(inp)])

def intent_help() -> None:
    """
        Desc:
            This is just a wrapper function for printing help
        return: None
    """
    
    print("""JAMSIE:     I can sure help! Here's a list of what I can do:\n
            You can greet me, and I will greet you right back!\n
            I can remember your name, simply say something along the lines of 'I want to be named' with your name\n
            I can help you out with a playlist on your system, just mention your playlist in any form like 'I want to see the artists in my playlist' or 'I want the longest song in my playlist' for example\n
            Just like right now, you can ask me for help, and I will display this very message!\n
            Since I possess extensive knowledge, I can answer a good majority of questions thanks to the dataset I was given, just say something like 'I want to ask a question' or 'I have an inquiry' and then ask your question!.            
            Last, but certainly not least, you can end our conversation. Like any other conversation, you just have to say a variation of bye. You can even say exit and our conversation will end.
          """)
    
    user_input = input("JAMSIE: Would you like to know more about anything? If so, ask more!. Otherwise simply refuse.\nYOU: ")

    user_input = lemmatizeString(string_preprocess(user_input))

    help_intent = intent(intents['help_intents'],user_input)


    asking = True
    while asking:
        # why cant python have syntatically good switch case...
        if help_intent == "stop":
            print("JAMSIE: The purpose of this is to end our conversation. Don't worry, I will still be here for the next time you need me. :D")
            asking = False
        elif help_intent == "name_retrieval":
            print("JAMSIE: To help provide personalisation to this conversation, I can remember your name. All you have to do is tell me your name, and I will remember it.\nFor example, this is me telling you my name, 'I am JAMSIE'.\nI will also be able to say your name at key points. You can try 'What's my name?'")
            asking = False
        elif help_intent == "greeting":
            print("JAMSIE: I love a good greeting. So all you need to do is say hello to me in any way you want and I will greet you right on back! :D")
            asking = False
        elif help_intent == "playlist":
            # flesh this out more when i figure out what to actually do with this
            print("JAMSIE: My main purpose is to help you with your playlist! We will keep a playlist together, in a relative directory to here. I can tell you information to your playlist, sort your playlist by certain factors like duration or song title. ")
            asking = False
        elif help_intent == "help":
            print("JAMSIE: Asking for help will just provide you with a simple description of each of my features. After this, you have the option to follow up asking more on any of the topics. This is actually how you got here, good job partner! :D")
            asking = False
        elif help_intent == "question":
            print("JAMSIE: I am absolutely brimming with knowledge. You can tell me that you wish to ask a question and I will start listening, after this you can ask your question and I will do my best to answer! :D")
            asking = False
        elif help_intent == "no":
            print("JAMSIE: No worries! Remember, if you ever need help, just ask :D")
            asking = False
        else:
            print("JAMSIE: Im sorry, I didn't understand what you asked. Would you like to ask again?")
            
            # help_intent = intent(intents['help_intents'],user_input)
                

def intent(intents: list[dict], user_input: str, thresh_ig = False) -> str:
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

    THRESHOLD = 0.2

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

    probs = vect_and_class.predict_proba([user_input]) # probabilities

    confidence =  probs.max() - probs.min()
    
    # print(f"confidence of incoming intent {round(confidence,4)}")
    # return the most likely / most confident, intent
    if thresh_ig == True: # Used to bypass confidence threshold (for example in the getName function, it wil be 0.5 confident and then recursively call and be 0.02 confident, which isnt relevant to the processing of the name)
        return predicted_intent[0]
    else:
        return predicted_intent[0] if confidence > THRESHOLD else None

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
   
    if compound > high_bound:
        return 1
    elif compound < low_bound:
        return -1
    else:
        return 0

def remove_stop_words(inp: list[str])-> list[str]:

    stop_words = set(stopwords.words('english'))

    tokenized_txt_stop = []

    for word in inp:
        if word not in stop_words:
            tokenized_txt_stop.append(word)

    return tokenized_txt_stop

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
    user_input = lemmatizeString(user_input)

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
    #print(f"Tags are {tags}")

    if grab_by_tag:
        group = []
        for tag in tags:
            if tag[1] == grab_by_tag:
                group.append(tag)
        return group
    else:
        return tags

def nameProcessor(inp: str) -> None:
    global user_name
    
    if intent(intents["name_intents"],inp,True) == "get_name":
        if user_name != None:
            print(f"JAMSIE: You are {user_name}, how could I forget you?")
            if len(user_hobbies) > 0:
                print("JAMSIE: You told me the following about your hobbies:\n")
                for elem in user_hobbies:
                    print("\t" + elem)
        else:
            print(f"JAMSIE: You havent told me your name. :(")
    elif intent(intents["name_intents"],inp,True) == "set_name":
        return getName(inp,0)
        
    else:
        print("JAMSIE: Sorry, while determining what you meant with your name, I got rather confused.")
   
def getName(inp: str, attempts: int) -> None:
    global user_name
    
    ALLOWED_ATTEMPS = 3 # this is a constant and must not be accessed

    intended_result = intent(intents["general_intents"],inp,True)

    # Names are somtimes miscalculated as VBN or RB in this POS set. So im going to grab the first RB or VBN found and process it as if it was a name

    try:
        find_nnp = POS(inp,"NNP")[0][0]
        if len(find_nnp) == 0:
            inp = POS(inp,"VBN")[0][0]
            if len(inp) == 0:
                inp = POS(inp,"RB")[0][0]
    except IndexError:
        pass # this is here allow an exception. I do this because proper NNP sentences will trigger this, thus we dont need to care. 
    
    
    if intended_result != "name_retrieval":
        intent_decider(intended_result, inp)
        return user_name

    if attempts >= ALLOWED_ATTEMPS:
        print("JAMSIE: I am sorry, I really struggled to catch your name, I do apologise.")
        return user_name 
    
    if len(inp.split(" ")) < 2:
        # A very bad bodge. But it works (singular words dont work well for identifying NNP for tags)
        inp = "please call me " + inp

    try:
    
        user_name_input = POS(inp,"NNP")[0][0]
        user_input = input(f"JAMSIE: So, you would like me to refer to you as {user_name_input}?\nYOU: ")
        intended_result = intent(intents["general_intents"],user_input,True)
        
        if intended_result != "yes_no": # Continue this to intercept the new intent of a user here 
            intent_decider(intended_result, inp)
            return user_name
        
        sentiment_input = sentiment(user_input)

        if sentiment_input <= 0:

            attempts += 1
            user_input = input(f"JAMSIE: What would you like me to call you?\nYOU: ")

            if len(user_input.split(" ")) < 2:
                # A very bad bodge. But it works (singular words dont work well for identifying NNP for tags)
                user_input = "please call me " + user_input

            return getName(user_input,attempts)

        else: 
            print(f"JAMSIE: Nice to meet you {user_name_input}")
            return user_name_input
        
    except IndexError as name_error:
        print("Sorry, I couldn't quite catch your name. I am only limited to english names.")
        attempts += 1
        getName(input(f"JAMSIE: What would you like me to call you?\nYOU: "),attempts)

def stp(inp: string) -> None: # could use NLG here if wanted/needed
    """
        Desc:
            This function is the small talk processor. It will grab the small talk and intent match it 
            to its corresponding intent dictionary. From this we can logically process the small talk conversation
        
        return: None
    """
    
    stp_intent = intent(intents['small_talk'],inp)

    if stp_intent == "weather":
        print("JAMSIE: I don't know what the weather is like. My developer was too lazy to give me that functionality. :(")
    
    elif stp_intent == "how":

        if user_name != None:
            emotion = sentiment(input(f"JANMSIE: I'm very well thank you {user_name}, how are you?\nYOU: "))
        else:
            emotion = sentiment(input(f"JANMSIE: I'm very well thank you, how are you?\nYOU: "))
        
        if emotion == 1:
            if user_name != None:
                print(f"JAMSIE: That's good! Im glad to hear that {user_name}")
            else:
                print(f"JAMSIE: That's good! Im glad to hear that")
        elif emotion == 0:
            if user_name != None:
                print(f"JAMSIE: I'm not sure how you are feeling {user_name}, im sorry")
            else:
                print("JAMSIE: I'm not sure how you are feeling, im sorry")
        elif emotion == -1:
            print(f"JAMSIE: I'm sorry to hear this. :(")
    
    elif stp_intent == "time":
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        if user_name != None:
            print(f"JAMSIE: According to my virtual watch, it is {current_time} {user_name}!")
        else:
            print(f"JAMSIE: According to my virtual watch, it is {current_time}!")
    elif stp_intent == "hobby":
        if user_name != None:
            user_hobby = input(f"JAMSIE: My hobbies include exploring Ethics in AI, programming, natural language processing, and chatbot development. I also have an interest in data science, machine learning, user experience design, and the intersection of linguistics and cognitive science.\n\tWhat are yours {user_name}?\nYOU:")
        else:
            user_hobby = input(f"JAMSIE: My hobbies include exploring Ethics in AI, programming, natural language processing, and chatbot development. I also have an interest in data science, machine learning, user experience design, and the intersection of linguistics and cognitive science.\n\tWhat are yours?\nYOU:")
        user_hobbies.append(string_preprocess(user_hobby))
    else:
        if user_name != None:
            print(f"JAMSIE: hmm, im not sure {user_name}")
        else:
            print("JAMSIE: hmm, im not sure") 

def intent_decider(intent: string, inp: string) -> None:
    global user_name
    """
        Desc:
            The purpose of this function is just to clean up code for the 
            god awful set of if statements that help the code logically 
            do stuff after an intent is found
        return: None
    """
    if intent == "stop":
        if user_name:
            print(f"JAMSIE: Goodbye {user_name}!")
        else:
            print("JAMSIE: Goodbye!")
        exit(0)

    else:
        # why cant python have syntatically good switch case...
        if intent == "name_retrieval":
            nameProcessor(inp)
            # user_name = getName(inp,0)
        elif intent == "greeting":
            greetings = ["Hello", "Hi", "Hey", "Howdy", "Greetings", "Salutations","Good day","Hey there"]
            print(f"JAMSIE: {greetings[randint(0,len(greetings)-1)]}. ")
            if user_name == None:

                name = string_preprocess(input(f" Oh no!, I don't know your name, what is your name?\nYOU: ")).lower()
                
                if len(name.split(" ")) < 2:
                    name = "i am " + name 
                user_name = nameProcessor(name)
            else:
                print(f"JAMSIE: {greetings[randint(0,len(greetings)-1)]} {user_name}. ")

        elif intent == "thanks":
            print("JAMSIE: Not a problem, glad to help! :D")
        elif intent == "transaction":
            print("transaction")
        elif intent == "void":
            print("JAMSIE: No worries, take your time!")
        elif intent == "help":
            intent_help()
        elif intent == "question":
            
            sub_process = True

            while sub_process:
                    
                if user_name != None:
                    question = input(f"JAMSIE: What would you like to ask me {user_name}?\nYOU: ")
                else:
                    question = input(f"JAMSIE: What would you like to ask me?\nYOU: ")
            
                answer = questionAnswer(qa_data,question)

                if answer == None:
                    
                    print("JAMSIE: Sorry, I couldn't understand your question.")
                    
                print(f"Answer: {answer}")
                sub_process = False

        elif intent == "small_talk":
            stp(inp)
        else:
            print("JAMSIE: I am unsure what you are asking of me, sorry. :(")

def con_exp(inp: str) -> str:
    """
        Desc: 
            This function works by changing the contracted words into their expanded counterparts.
            With thanks to the person who made the dictionary, the link to the data can be found
            above the dictionary.
        return: string
    """

    tokens = nltk.word_tokenize(inp)
    expansions = [contractions.get(word, word) for word in tokens]
    expanded_text = ' '.join(expansions)
    return expanded_text

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

    vectorized_strings = vectorizer.fit_transform(data) # vectorize the questions

    vectorized_new_string = vectorizer.transform([inp]) # vectorize the answers

    cosine_similarities = cosine_similarity(vectorized_new_string, vectorized_strings) # perform the cosine similarity between the two vectors given

    if np.sum(cosine_similarities) == 0: # Input is dissimilar to everything given
        return None

    return np.argmax(cosine_similarities)

def questionAnswer(qa_package: list[tuple],question: string):

    questions = [x[0] for x in qa_package]

    sim_index = similirityMatching(questions,question)

    return qa_package[sim_index][1] if sim_index != None else None

# Program loop

print(hi_string)
print("-"*len(hi_string))

while running :
   
    if already_asked:
        prompt = string_preprocess(con_exp(input("\nJAMSIE: What else can I help you with?\n\nYOU: ").lower()))
    else:
        already_asked = True
        prompt = string_preprocess(con_exp(input("\nJAMSIE: What can I help you with?\n\nYOU: ").lower()))

    user_intent = intent(intents["general_intents"],prompt)

    intent_decider(user_intent,prompt)




# PlaylistManager.sortSongs()




## HELP DOCUMENTATION

# https://www.nltk.org/book/ch05.html

# https://www.nltk.org/api/nltk.sentiment.vader.html

# https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python#:~:text=Cosine%20Similarity%20computes%20the%20similarity,the%20cosine%20similarity%20is%201