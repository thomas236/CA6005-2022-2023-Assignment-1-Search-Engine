"""
Dublin City University
MSc Computer Science (Artificial Intelligence)
CA6005 2022-2023 Mechanics of Search
Assignment 1: Search Engine
Student name: Thomas Sebastian
Student ID: 21250103 (University of Galway, Ireland)
email address: thomas.sebastian3@mail.dcu.ie
"""
# Create folders necessary for storing parsed and processed text. Delete pre-existing folders
import os
folder_paths = ["cranfieldDocs", "preprocessed_cranfieldDocs", "cranfieldQueries", "preprocessed_cranfieldQueries"]
for folder_path in folder_paths:
    # Check if folder exists
    if os.path.exists(folder_path):
        # Delete folder and its contents
        os.system("rm -rf " + folder_path)
        # Create folder
        os.makedirs(folder_path)
    # Create a directory to store the output files
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)       

# Delete 'resultsCosineSimilarity.txt' file
if os.path.exists("resultsCosineSimilarity.txt"):
    os.remove("resultsCosineSimilarity.txt")

# Delete 'resultsBM25.txt' file
if os.path.exists("resultsBM25.txt"):
    os.remove("resultsBM25.txt")    
    
# Delete 'resultsGensim.txt' file
if os.path.exists("resultsGensim.txt"):
    os.remove("resultsGensim.txt")  
    
# Delete Evaluation files if they exist
if os.path.exists("cosine_eval.txt"):
    os.remove("cosine_eval.txt")

if os.path.exists("bm25_eval.txt"):
    os.remove("bm25_eval.txt")
    
if os.path.exists("gensim_eval.txt"):
    os.remove("gensim_eval.txt")
    
import os
import xml.etree.ElementTree as ET

# Create a directory to store the query files
os.makedirs("cranfieldQueries", exist_ok=True)

# Load the Query XML file
tree = ET.parse("cran.qry.xml")
root = tree.getroot()

# Extract the queries with tag "top"
queries = root.findall("top")

# Process each query, specifically information in tags "num" and "title"
# which contain the query number and the text for query
for query in queries:
    # Extract the query number and title
    query_num = query.find("num").text
    query_title = query.find("title").text
    
    # Save the query title to a file for pre-processing
    filename = os.path.join("cranfieldQueries", f"{query_num}")
    with open(filename, "w") as file:
        file.write(query_title)
        
        
from bs4 import BeautifulSoup
import os

# Load the cran.all.1400 XML fil
with open('cran.all.1400.xml', 'r') as file:
    data = file.read()

# Parse the XML file using Beautiful Soup by creating an BS object
soup = BeautifulSoup(data, 'xml')

# Extract the docno and text elements from each document and save to a new file
# with tag "doc"
for doc in soup.find_all('doc'):
    docno = doc.docno.string.strip()

    # Create a new file for the document and write text into the same file name
    with open(f'cranfieldDocs/{docno}', 'w') as file:
        file.write(str(doc))
        
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup as BS

import string
from nltk.tokenize import word_tokenize


#file path for original Cranfield docs
raw_file_path = 'cranfieldDocs'
#file path for pre processed docs in cranfield documennts set
preprocessed_file_path = 'preprocessed_cranfieldDocs'

#create folder for preprocessed files
if os.path.isdir(preprocessed_file_path):
    pass
else:
    os.mkdir(preprocessed_file_path)    
    
#Get list of all docs in the Cranfield database
file_names = os.listdir(raw_file_path)

import os
import re
from nltk.stem import WordNetLemmatizer

# Create a directory to store the preprocessed queryfiles
os.makedirs("preprocessed_cranfieldQueries", exist_ok=True)

# Load the stop words
stop_words = set(stopwords.words('english'))

# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Process each file in the cranfieldQueries folder
for filename in os.listdir("cranfieldQueries"):
    # Read the contents of the file
    with open(os.path.join("cranfieldQueries", filename), 'r') as file:
        contents = file.read()
    
    # Preprocess the contents
    contents = contents.replace(".", "")
    contents = re.sub('[^a-zA-Z0-9\n\.]', ' ', contents)
    contents = word_tokenize(contents.lower())
    contents = [word for word in contents if word not in stop_words]
    contents = [lemmatizer.lemmatize(word) for word in contents]
    contents = [stemmer.stem(word) for word in contents]
    
    # Save the preprocessed contents to a file with the same filename in the preprocessed_cranfieldQueries folder
    with open(os.path.join("preprocessed_cranfieldQueries", filename), 'w') as file:
        file.write(" ".join(contents))
        
# function to tokenize. Parameters are a BS text and tag
def tokenize(file_soup_object, tag):
    #extract informatio between TREC tags
    TREC_data = file_soup_object.findAll(tag)
    
    # convert to string and remove tag related information from the string
    TREC_data = ''.join(map(str, TREC_data))
    TREC_data = TREC_data.replace(tag," ")
    
    # Convert string data to lower case, remove punctuations, stop wrods
    TREC_data = TREC_data.lower()
    TREC_data = TREC_data.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(TREC_data)
   
    # Tokenize  text
    tokens = word_tokenize(TREC_data)

    # Remove stop words
    # get a list of stopwords to be used for pre-processing the Cranfield dataset
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in tokens if word not in stop_words]

    # Stem the tokens
    # Initialise a Porter Stemmer Object to Create a Tree/ Stem Model to scan through documents
    stemmer = PorterStemmer()
    stem_tokens = [stemmer.stem(word) for word in clean_tokens]

    # Convert list of tokens to string
    clean_stem_tokens = ' '.join(stem_tokens)
    
    return clean_stem_tokens


#Pre-Process raw files and save it in new folder "preprocessed_cranfieldDocs"
#file path for original Cranfield docs
raw_file_path = 'cranfieldDocs'
#file path for pre processed docs
preprocessed_file_path = 'preprocessed_cranfieldDocs'
#scan through each file name
for file in file_names:
    raw_file = raw_file_path+ "/" + file
    preprocessed_file = preprocessed_file_path + "/" + file
    
    with open(raw_file) as r_file:
        with open(preprocessed_file, 'w') as p_file:
            # read information in each raw cranfield document
            file_information = r_file.read()
            #BeautifulSoup object to extract information between TREC tags
            file_soup_object = BS(file_information)
            
            #extract docnofrom each raw file
            doc_no = tokenize(file_soup_object, 'docno')
            
            #extract title from each raw file
            title = tokenize(file_soup_object, 'title')
            
            #extract author number from each raw file
            author = tokenize(file_soup_object, 'author')
                        
            #extract bibliography number from each raw file
            biblio = tokenize(file_soup_object, 'biblio')
                      
            #extract text number from each raw file
            text = tokenize(file_soup_object, 'text')
            
            # save tokeized infomation into new files in pre-procecessed directory
            p_file.write(text)
            p_file.write(" ")
        p_file.close()
    r_file.close()

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# iteration and run_id values are not relevant but necessary
#for dyn_eval
iteration = "Q0"
run_id = "140"

#return top 100 documents
rank = 100

# Define paths to the preprocessed files
queries_path = "preprocessed_cranfieldQueries"
docs_path = "preprocessed_cranfieldDocs"

# Get file names of the queries and docs
# Get results of all files
queries_filenames = sorted(os.listdir(queries_path)) 
docs_filenames = sorted(os.listdir(docs_path))

# Load the preprocessed queries and docs
queries = []
for filename in queries_filenames:
    with open(os.path.join(queries_path, filename), 'r') as file:
        queries.append(file.read())
        
docs = []
for filename in docs_filenames:
    with open(os.path.join(docs_path, filename), 'r') as file:
        docs.append(file.read())

# Compute the TF-IDF matrix for the queries and docs
vectorizer = TfidfVectorizer()
tfidf_queries = vectorizer.fit_transform(queries)
tfidf_docs = vectorizer.transform(docs)

# Calculate the cosine similarity between each query and each doc
similarity_matrix = cosine_similarity(tfidf_queries, tfidf_docs)

# Write the results to a text file
with open("resultsCosineSimilarity.txt", "w") as f:
    for i, query_similarities in enumerate(similarity_matrix):
        query_id = os.path.splitext(queries_filenames[i])[0]
        top_matches = sorted(enumerate(query_similarities), key=lambda x: x[1], reverse=True)[:rank]
        for j, (doc_index, similarity) in enumerate(top_matches):
            doc_id = os.path.splitext(docs_filenames[doc_index])[0]
            f.write(f"{query_id} {iteration} {doc_id} {j+1} {similarity:.5f} {run_id}\n")
            
with open("resultsCosineSimilarity.txt", "r") as f:
    lines = f.readlines()

with open("resultsCosineSimilarity.txt", "w") as f:
    for line in lines:
        parts = line.split()
        # remove spaces in the first column
        parts[0] = parts[0].replace(" ", "")  
        f.write(" ".join(parts) + "\n")
        
# Open the file in read mode and read the lines
with open("resultsCosineSimilarity.txt", "r") as f:
    lines = f.readlines()

# Sort the lines in ascending order of the first column
sorted_lines = sorted(lines, key=lambda line: int(line.split()[0]))

# Open the file in write mode and write the sorted lines
with open("resultsCosineSimilarity.txt", "w") as f:
    f.writelines(sorted_lines)
    
    
from rank_bm25 import BM25Okapi
import os


# iteration and run_id values are not relevant but necessary
#for dyn_eval
iteration = "Q0"
run_id = "140"

# Path to directories containing preprocessed files
query_dir = "preprocessed_cranfieldQueries"
doc_dir = "preprocessed_cranfieldDocs"

# Collect all preprocessed queries and documents
queries = []
for filename in os.listdir(query_dir):
    with open(os.path.join(query_dir, filename), "r") as f:
        queries.append(f.read().split())

docs = []
for filename in os.listdir(doc_dir):
    with open(os.path.join(doc_dir, filename), "r") as f:
        docs.append(f.read().split())

# Create BM25 object with default parameters
bm25 = BM25Okapi(docs, k1=1.2, b=0.75)

# Compute similarities between queries and documents
similarities = []
for query in queries:
    query_scores = bm25.get_scores(query)
    similarities.append([(score, i) for i, score in enumerate(query_scores)])

# Sort similarities in descending order and save top 100 ranks to file
with open("resultsBM25.txt", "w") as f:
    for i, sim in enumerate(similarities):
        ranked_sim = sorted(sim, reverse=True)[:100] # only top 100 ranks
        for rank, (score, doc_index) in enumerate(ranked_sim):
            doc_id = os.path.splitext(os.listdir(doc_dir)[doc_index])[0]
            f.write(f"{os.path.splitext(os.listdir(query_dir)[i])[0]} {iteration} {doc_id} {rank+1} {score:.5f} {run_id}\n")
            
with open("resultsBM25.txt", "r") as f:
    lines = f.readlines()

with open("resultsBM25.txt", "w") as f:
    for line in lines:
        parts = line.split()
        parts[0] = parts[0].replace(" ", "")  # remove spaces in the first column
        f.write(" ".join(parts) + "\n")
        
# Open the file in read mode and read the lines
with open("resultsBM25.txt", "r") as f:
    lines = f.readlines()

# Sort the lines in ascending order of the first column
sorted_lines = sorted(lines, key=lambda line: int(line.split()[0]))

# Open the file in write mode and write the sorted lines
with open("resultsBM25.txt", "w") as f:
    f.writelines(sorted_lines)
    
import os
from gensim import corpora, similarities


# iteration and run_id values are not relevant but necessary
#for dyn_eval
iteration = "Q0"
rank =100
run_id = "140"

# Path to directories containing preprocessed files
query_dir = "preprocessed_cranfieldQueries"
doc_dir = "preprocessed_cranfieldDocs"

# Collect all preprocessed queries and documents
queries = []
for filename in os.listdir(query_dir):
    with open(os.path.join(query_dir, filename), "r") as f:
        queries.append(f.read().split())

docs = []
for filename in os.listdir(doc_dir):
    with open(os.path.join(doc_dir, filename), "r") as f:
        docs.append(f.read().split())

# Build dictionary from documents
dictionary = corpora.Dictionary(docs)

# Convert documents into bag-of-words format and index them
corpus = [dictionary.doc2bow(doc) for doc in docs]
index = similarities.MatrixSimilarity(corpus)

# Process queries and retrieve top 100 similar documents for each query
similarities = []
for query in queries:
    query_vec = dictionary.doc2bow(query)
    sims = index[query_vec]
    similarities.append([(score, i) for i, score in enumerate(sims)])

    
# Sort similarities in descending order and save top 100 ranks to file
with open("resultsGensim.txt", "w") as f:
    for i, sim in enumerate(similarities):
        ranked_sim = sorted(sim, reverse=True)[:100] # only top 100 ranks
        for rank, (score, doc_index) in enumerate(ranked_sim):
            doc_id = os.path.splitext(os.listdir(doc_dir)[doc_index])[0]
            f.write(f"{os.path.splitext(os.listdir(query_dir)[i])[0]} {iteration} {doc_id} {rank+1} {score:.5f} {run_id}\n")

with open("resultsGensim.txt", "r") as f:
    lines = f.readlines()

with open("resultsGensim.txt", "w") as f:
    for line in lines:
        parts = line.split()
        parts[0] = parts[0].replace(" ", "")  # remove spaces in the first column
        f.write(" ".join(parts) + "\n")
        
# Open the file in read mode and read the lines
with open("resultsGensim.txt", "r") as f:
    lines = f.readlines()

# Sort the lines in ascending order of the first column
sorted_lines = sorted(lines, key=lambda line: int(line.split()[0]))

# Open the file in write mode and write the sorted lines
with open("resultsGensim.txt", "w") as f:
    f.writelines(sorted_lines)
    
import os
# Execute dyn_eval to run Dyneval and extract desired metrics MAP, P@5, and NDCG
os.system('./dyneval qrel resultsCosineSimilarity.txt | grep -E "(map|P_5 |ndcg)" > cosine_eval.txt')
os.system('./dyneval qrel resultsBM25.txt | grep -E "(map|P_5 |ndcg)" > bm25_eval.txt')
os.system('./dyneval qrel resultsGensim.txt | grep -E "(map|P_5 |ndcg)" > gensim_eval.txt')