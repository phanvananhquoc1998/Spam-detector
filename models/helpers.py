import itertools
import re
import nltk

def remove_punctuations(my_str):
    punctuations = '''!()-[]{};:'"\,./?@#$%^&@*_~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def remove_appostophes(sentence):
    APPOSTOPHES = {"s": "is", "re": "are", "t": "not",
                   "ll": "will", "d": "had", "ve": "have", "m": "am"}
    words = nltk.tokenize.word_tokenize(sentence)
    final_words = []
    for word in words:
        broken_words = word.split("'")
        for single_words in broken_words:
            final_words.append(single_words)
    reformed = [APPOSTOPHES[word]
                if word in APPOSTOPHES else word for word in final_words]
    reformed = " ".join(reformed)
    return reformed


def clean_data(sentence):
    # removing web links
    s = [re.sub(r'http\S+', '', sentence.lower())]
    # removing words like gooood and poooor to good and poor
    s = [''.join(''.join(s)[:2] for _, s in itertools.groupby(s[0]))]
    # removing appostophes
    s = [remove_appostophes(s[0])]
    # removing punctuations from the code
    s = [remove_punctuations(s[0])]
    return s[0]
