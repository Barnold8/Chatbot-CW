## This is just a simple file to process strings found online to format nicely into intents
## I can then copy the terminal output and paste it into my intents data and not worry about punctuation or different casings
import re

def noPunc(string: str) -> str:

    string_proc = ""
    for char in string:
        if(bool(re.match('^[a-zA-Z ]*$',char))==True):
            string_proc += char

    return string_proc

data = [ ## Put strings in here that you want to be processed

]

data_processed = [x.lower() for x in data]

data_processed = [noPunc(x) for x in data_processed]



print(data_processed)
