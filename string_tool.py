## This is just a simple file to process strings found online to format nicely into intents
## I can then copy the terminal output and paste it into my intents data and not worry about punctuation or different casings
import re

def noPunc(string: str) -> str:

    string_proc = ""
    for char in string:
        if(bool(re.match('^[a-zA-Z ]*$',char))==True):
            string_proc += char

    return string_proc

## Put strings in here that you want to be processed

data = [
    "I'm curious about how to properly ask a question.",
    "Can you guide me on the best way to formulate a question?",
    "I want to know the right way to ask a question.",
    "Could you share tips on how to ask questions effectively?",
    "I'm interested in learning how to frame questions in the right manner.",
    "How do you suggest I ask a question to get the best response?",
    "I'd appreciate advice on how to articulate my questions clearly.",
    "What's the proper etiquette for asking questions?",
    "Can you provide guidance on when and how to ask a question?",
    "I'm looking for tips on communicating my queries effectively.",
    "Could you help me with strategies for asking questions in a respectful way?",
    "I want to understand the best practices for seeking information through questions.",
    "Is there a recommended way to phrase questions for better understanding?",
    "I'm curious about the social norms for asking questions.",
    "Is there a polite way to seek information by asking questions?",
]




data_processed = [x.lower() for x in data]

data_processed = [noPunc(x) for x in data_processed]



print(data_processed)
