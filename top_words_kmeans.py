from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import os
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove numbers, punctuation, and stopwords
    stopwords_english = set(stopwords.words('english'))
    stopwords_french = set(stopwords.words('french'))

    # Combine stopwords from both languages into a single set
    stopwords_combined = stopwords_english.union(stopwords_french)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords_combined]
    
    return filtered_tokens

def read_documentation(): 
    folder_path = '/home/grisel/Documents/internship/venv/files_dataset/export/'
    #document_1 = '/home/grisel/Documents/internship/venv/files_dataset/export/010046797.pdf'
    #document_2 = '/home/grisel/Documents/internship/venv/files_dataset/export/010017940.pdf'
    #document_3 = '/home/grisel/Documents/internship/venv/files_dataset/export/010038507.pdf'
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
    documents = []
    document_names = [] 
    for pdf_path in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_path)
            documents.append(text)
            document_names.append(os.path.basename(pdf_path))  # Get only the filename without path
        except Exception as e:
            print(f"Error reading '{pdf_path}': {e}. Skipping this document.")
            
    #pdf_files = [document_1]
    #documents = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_files]
    return(documents, document_names)

def document_embeddings(documents):
    # Preprocess documents
    preprocessed_documents = [preprocess_text(doc) for doc in documents]
    
    # Train Word2Vec model
    model = Word2Vec(preprocessed_documents, vector_size=100, window=5, min_count=1, workers=4)
    
    # Aggregate word embeddings for each document
    document_vectors = []
    for doc in preprocessed_documents:
        doc_vector = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
        document_vectors.append(doc_vector)
    
    return np.array(document_vectors), model
    #return (document_vectors)

def top_words_dataframe(document_vectors, document_names, model, num_words=66):
    # Cluster documents based on their embeddings
    kmeans = KMeans(n_clusters=num_words, random_state=42)
    kmeans.fit(document_vectors)
    
    # Get cluster centroids
    centroids = kmeans.cluster_centers_
    
    # Get top words based on centroids
    top_words = []
    for centroid in centroids:
        # Find closest word vector to centroid
        closest_word_index = np.argmin(np.linalg.norm(document_vectors - centroid, axis=1))
        top_words.append(model.wv.index_to_key[closest_word_index])
    
    
    df_top_words = pd.DataFrame({
        'Document': range(1, len(document_names) + 1),
        'Document Name': document_names,
        'Top Words': top_words
    })
    
    return df_top_words

documents, document_names = read_documentation()
document_vectors, model = document_embeddings(documents)
df = top_words_dataframe(document_vectors, document_names, model)
df.to_csv('key_words_kmeans.csv', index=False)
print("Top words with kmeans saved to 'key_words_kmeans.csv'")