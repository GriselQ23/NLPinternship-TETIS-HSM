from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


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


def stopword_removal(): 
    # Download NLTK stopwords 
    #nltk.download('stopwords')

    # Get French stopwords from NLTK
    stop_words = list(stopwords.words('french'))

    # Create a TfidfVectorizer with NLTK stopwords
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Fit the vectorizer to the data and transform the documents
    documents, document_names = read_documentation() 
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    return(tfidf_matrix, feature_names)


def key_words():
    #Define the specific words 
    specific_words = ['avoine', 'riz', 'millet', 'seigle', 'sorgho', 'orge', 'artichaut',
    'asperges', 'choux', 'choux-fleurs et brocolis', 'salades', 'epinard', 'chicoré', 
    'concombre', 'aubergine', 'tomates', 'pastèques', 'melons', 'citrouille', 'courge et calebasses',
    'carotte','oignons', 'taro', 'navets', 'ail', 'poireaux', 'arachides',
    'graine de ricin', 'lin', 'moutarde','guizotia abyssinie', 'colza', 'carthame', 'sésame', 'tournesol',
    'manioc', 'pomme de terre', 'pomme', 'patate douce', 'igname', 'haricots', 'pois', 'soja',
    'coton', 'kiwis', 'agrumes', 'ananas', 'goyave', 'grenade', 'kaki', 'papaye', 'mangue', 
    'anacardier','noix de cajou', 'noix de pécan','pacanier', 'pistachier', 'jachère',
    'savane arborée', 'savane arbustive', 'savane herbacée et prairie', 'forêt à feuilles persistantes', 'brousse tigrée', 'sol nu',
    'bâtiment', 'route', 'eau']

    # Initialize a list to store TF-IDF scores of specific words
    specific_words_tfidf = []

    #Get the tfidf_matrix and the feature_names
    tfidf_matrix, feature_names = stopword_removal()

    # Find indices of specific words in feature_names array
    word_indices = {word: idx for idx, word in enumerate(feature_names) if word in specific_words}

    # Get document texts and names
    documents, document_names = read_documentation()

    # Loop over each document
    for i, doc in enumerate(documents):
        tfidf_scores = tfidf_matrix[i, :].toarray().flatten()
        doc_specific_words_tfidf = []
        # Find TF-IDF scores for specific words
        for word in specific_words:
            if word in word_indices:
                word_index = word_indices[word]
                tfidf_score = tfidf_scores[word_index]
                doc_specific_words_tfidf.append(tfidf_score)
            else:
                doc_specific_words_tfidf.append(0)  # If word not found, set TF-IDF score to 0
        specific_words_tfidf.append(doc_specific_words_tfidf)

    # Create DataFrame with document indices and TF-IDF scores of specific words
    df_specific_words_tfidf = pd.DataFrame(specific_words_tfidf, columns=specific_words)
    df_specific_words_tfidf.insert(0, 'Document', range(1, len(documents)+1))
    df_specific_words_tfidf.insert(1, 'Document Name', document_names)

    return(df_specific_words_tfidf)

def top_words():
    # Read documents
    documents, document_names = read_documentation()

    #Get the tfidf matrix and feature names
    tfidf_matrix, feature_names = stopword_removal()

    # Sum the TF-IDF scores for each word across all documents
    word_scores = tfidf_matrix.sum(axis=0).A1

    # Create a DataFrame to store word scores
    df_word_scores = pd.DataFrame({'Word': feature_names, 'Score': word_scores})

    # Sort words by their TF-IDF scores
    df_word_scores = df_word_scores.sort_values(by='Score', ascending=False)

    # Get the top 66 words
    top_words = df_word_scores.head(66)['Word'].tolist()

    # Initialize a list to store TF-IDF scores of top words
    top_words_tfidf = []

    # Find indices of top words in feature_names array
    word_indices = {word: idx for idx, word in enumerate(feature_names) if word in top_words}
    # Loop over each document
    for i, doc in enumerate(documents):
        tfidf_scores = tfidf_matrix[i, :].toarray().flatten()
        doc_top_words_tfidf = []
        # Find TF-IDF scores for top words
        for word in top_words:
            if word in word_indices:
                word_index = word_indices[word]
                tfidf_score = tfidf_scores[word_index]
                doc_top_words_tfidf.append(tfidf_score)
            else:
                doc_top_words_tfidf.append(0)  # If word not found, set TF-IDF score to 0
        top_words_tfidf.append(doc_top_words_tfidf)

    # Create DataFrame with document indices and TF-IDF scores of top words
    df_top_words_tfidf = pd.DataFrame(top_words_tfidf, columns=top_words)
    df_top_words_tfidf.insert(0, 'Document', range(1, len(documents)+1))
    df_top_words_tfidf.insert(1, 'Document Name', document_names)

    return df_top_words_tfidf  
#df = key_words()
#df.to_csv('specific_words_tfidf_scores.csv', index=False)
#print("TF-IDF scores of specific words saved to 'specific_words_tfidf_scores.csv'")

df = top_words()
df.to_csv('key_words_tfidf_scores.csv', index=False)
print("TF-IDF scores of key words saved to 'key_words_tfidf_scores.csv'")

"the, plus, del, of, afrique, ouest, and, in, entre, cette, comme, climatiques, climatique, donnes, 10, 
annes, etre pluies, tchad, bassins  periode, figure, fig, autres,  "