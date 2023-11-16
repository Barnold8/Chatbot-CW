import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('words')

# Load the English words set from NLTK
english_words = set(nltk.corpus.words.words())

# Custom corrections dictionary
custom_corrections = {"nam": "name"}  # Add more custom corrections as needed

# Function for spelling correction with custom corrections
def correct_spelling(text):
    words = nltk.word_tokenize(text)
    corrected_words = [custom_corrections.get(word.lower(), word) if word.lower() in custom_corrections or not word.isalpha() else suggest_correction(word) for word in words]
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Function to suggest a correction for a misspelled word
def suggest_correction(word):
    suggestions = [w for w in english_words if nltk.edit_distance(word, w) <= 1]
    return min(suggestions, key=lambda x: nltk.edit_distance(word, x)) if suggestions else word

# Example usage
text_with_errors = "I lovee natural languaage procesing. nam is misspelled."
corrected_text = correct_spelling(text_with_errors)

print("Original Text:", text_with_errors)
print("Corrected Text:", corrected_text)
