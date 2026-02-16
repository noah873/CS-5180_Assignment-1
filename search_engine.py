#-------------------------------------------------------------
# AUTHOR: Noah Loke
# FILENAME: search_engine.py
# SPECIFICATION: processes a corpus of acquired documents,
# then ranks them against a query
# FOR: CS 5180- Assignment #1
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

# ---------------------------------------------------------
#Importing some Python libraries
# ---------------------------------------------------------
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')
from nltk import word_tokenize

import warnings
warnings.filterwarnings('ignore')

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------
# --> add your Python code here

print("\nDocuments:", documents, end='\n\n')

# ---------------------------------------------------------
# Instantiate CountVectorizer informing 'word' as the analyzer, Porter stemmer as the tokenizer, stop_words as the identified stop words,
# unigrams and bigrams as the ngram_range, and binary representation as the weighting scheme
# ---------------------------------------------------------
# --> add your Python code here

class StemTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]

vectorizer = CountVectorizer(analyzer='word', tokenizer=StemTokenizer(), stop_words='english', ngram_range=(1,2), binary=True)

# ---------------------------------------------------------
# Fit the vectorizer to the documents and encode the them
# ---------------------------------------------------------
# --> add your Python code here

vectorizer.fit(documents)
document_matrix = vectorizer.transform(documents)

# ---------------------------------------------------------
# Inspect vocabulary
# ---------------------------------------------------------
print("Vocabulary:", vectorizer.get_feature_names_out().tolist(), end='\n\n')

# ---------------------------------------------------------
# Fit the vectorizer to the query and encode it
# ---------------------------------------------------------
# --> add your Python code here

query = "I love dogs"
query_vector = vectorizer.transform([query])

# ---------------------------------------------------------
# Convert matrices to plain Python lists
# ---------------------------------------------------------
# --> add your Python code here

doc_vectors = document_matrix.toarray().tolist()
query_vector = query_vector.toarray().tolist()[0]

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------

scores = []
# --> add your Python code here

for i in range(len(doc_vectors)):
    score = 0
    for a, b in zip(doc_vectors[i], query_vector):
        score += a * b
    scores.append([i+1, score])

# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------

ranking = sorted(scores, key=lambda x: x[1], reverse=True)
# --> add your Python code here

print("*** Sorted Ranking ***")
for i in range(len(ranking)):
    print(str(i+1) + ". ", end='')
    print("Document", ranking[i][0], end='')
    print(", Score:", ranking[i][1])
