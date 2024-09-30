import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:

    def __init__(self, text, nomenclature):    
        self.text = text
        self.nomenclature = nomenclature
    
    def read_csv(self, file_name):
        words = []
        with open(file_name, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                for word in row:
                    if word:  # Check if word is not empty
                        words.append(word)
        return words

    def compute_tf_idf_for_specific_words(self, preprocessed_text, specific_words, document_name='Document_1'):
        # Convert the preprocessed text into a single string
        document = ' '.join(preprocessed_text)

        # Initialize Vectorizer with specific words
        tfidf_vectorizer = TfidfVectorizer(vocabulary=specific_words)

        # Fit and transform the document
        tfidf_matrix = tfidf_vectorizer.fit_transform([document])

        # Get feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Get TF-IDF scores
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Create a DataFrame to store the TF-IDF scores
        df = pd.DataFrame()

        # Add document name to DataFrame
        df[document_name] = [document_name]

        # Add feature names and corresponding TF-IDF scores to DataFrame
        for feature, score in zip(feature_names, tfidf_scores):
            df[feature] = score

        return df

    def computation(self):
        specific_words = self.read_csv(self.nomenclature) 
        df = self.compute_tf_idf_for_specific_words(self.text, specific_words)
        return df

