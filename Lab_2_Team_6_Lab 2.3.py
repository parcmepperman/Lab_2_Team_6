from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import re


# find text using tokens
def find_tokens(text):
    return re.findall('[a-z]+', text.lower())


# open the text file, read the text

file_text = open('aesop.txt', 'r').read()

# call the function fn_tokens and store the return value in WORDS_input
input_txt = find_tokens(open('aesop.txt').read())

# lemmatize the tokens
print("\n")
print("The lemmatized text is......................................................................")
lem = []
lemmatizer = WordNetLemmatizer()
for word in input_txt:
    print(lemmatizer.lemmatize(word))

# Find the bi-gram
input_list = []
for WORD in input_txt:
    input_list = input_list+[WORD]


#Bi-gram frequency using Counter from the collections library
print("\n")
print("The TOP 5 bi-gram words in text file are...................................")
frequencies_counter = Counter([])
bi_grams = ngrams(input_txt, 2)
frequencies_counter += Counter(bi_grams)


#Top five bi-gram words
print("\n")
delete_list = []
for i in range(0, 5):
    delete_list.append(" ".join(re.findall("[a-zA-Z]+", str(frequencies_counter).split(":")[i])))
print(delete_list)

#Finding all the sentences with those most repeated bi-grams
print("\n")
print("This is a list of the sentences with that contains the TOP 5 bi-gram words...........................")
lines = {}
for line in file_text.split("."):
    for word in delete_list:
        if word in line:
            if line in lines:
                pass
            else:
                lines[line] = ""
result = list()
for line in lines:
    result.append(line+".")
print(result)
