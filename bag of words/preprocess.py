import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import os
import spacy

class Preprocess:

    def __init__(self, dataset_path):
        
        self.dataset_path = dataset_path

    def extract_text_from_pdf(self, pdf_path):
        text = ''
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def text_preparation(self,text):
        # Load the English language model
        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = 2000000  # Set max_length limit to 2,000,000 characters
        # Tokenization, stop words removal, and lemmatization
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return tokens
 
    
    def preprocess(self):
        text = self.extract_text_from_pdf(self.dataset_path)
        tokens = self.text_preparation(text)
        return(tokens)
   