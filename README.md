# NLPinternship-TETIS-HSM 🌍📚

## Project Overview

This project focuses on extracting textual information from scientific documents 📄, identifying important segments ✂️, and classifying them 🏷️ to better understand Land Use and Land Cover Changes (LULCC) in West Africa. By systematically processing and analyzing scientific literature, we aim to improve our understanding of land cover dynamics and support better-informed decisions for managing land use in regions facing rapid environmental changes, such as the Sahel.

### Motivation
Global warming and rapid changes in land use are having a significant impact on water resources and ecosystems in West Africa 🌡️🌱. Understanding the drivers of these changes—climate shifts, population growth, and agricultural expansion—is crucial for effective management and adaptation strategies. This project is part of the broader CECC (Cycle de l’eau et changement climatique) initiative, which aims to anticipate future changes in water resources due to climate change 💧.

### Objective
The core research question is:
**How can we effectively classify and analyze textual information related to Land Use and Land Cover Change (LULCC) in West Africa to better understand the temporal and quantitative dynamics driving these changes?**

Our aim is not only to understand these changes but also to improve the methods for labeling and analyzing scientific text data, providing valuable insights for land use decision-making 🌍🧠.

## Methodology and Procedure

1. **Corpus Creation**:
   - A custom corpus was built from 69 English-language scientific articles, selected using keywords related to land use, land cover, reforestation, deforestation, urbanization, and agriculture in West Africa 📑.
   - The nomenclature (dictionary of terms) was developed using JECAM standards and expanded with AGROVOC (FAO) terms 📖.

2. **Preprocessing**:
   - PDF documents were converted to TEI format using GROBID, preserving document structure and reducing noise 🛠️.
   - Text was segmented into sentences using regular expressions, and relevant segments were identified using NLP techniques (tokenization, stemming, lemmatization with SpaCy) 🧩.

3. **Labeling**:
   - Segments were manually labeled by experts into three categories: non-relevant, potentially relevant, and definitively relevant 🏷️.
   - Iterative expert review improved labeling accuracy and consistency 👩‍🔬👨‍🔬.

4. **Classification and Extraction**:
   - After segmentation and segment extraction, we use BERT for classification 🤖 and Named Entity Recognition (NER) for extracting important data such as geo-localization, numeric values, geo positions, and more 🌍🔢📍.
   - Supervised machine learning models (SVM, BERT/RoBERTa) are trained to classify segments, using TF-IDF for feature extraction and grid search for hyperparameter optimization 🤖.
   - Evaluation metrics include accuracy, precision, recall, and F1-score, with a focus on maximizing recall to capture all relevant information 📊.

5. **Information Extraction**:
   - Named Entity Recognition (NER) is performed using SpaCy, Regex, and EntityRuler to extract quantitative and qualitative entities (measurements, dates, locations, land cover terms, change processes) 🔍.

## Technologies and Libraries Used

- **Natural Language Processing (NLP)**: SpaCy, Regex, GROBID 🧠
- **Machine Learning**: scikit-learn (SVM), HuggingFace Transformers (BERT, RoBERTa) 🤖
- **Data Processing**: pandas, numpy, PyPDF2 📊
- **Visualization and Analysis**: matplotlib, seaborn 📈
- **Entity Extraction**: SpaCy NER, custom vocabularies 🔎
- **mlxtend**: Frequent pattern mining
- **tqdm**: Progress bars

## Project Structure and Purpose

### 1. Dataset Folder
- **create_dataset.ipynb**: Processes raw TEI files and generates structured datasets in CSV format 🗂️.
- **dataset_label.csv**: Labeled data for supervised learning 🏷️.
- **dataset_unlabel.csv**: Unlabeled data for unsupervised/semi-supervised tasks 🕵️‍♂️.
- **nomenclature_v3.csv**: Dictionary of terms for consistent labeling and analysis 📖.

### 2. Analysis Folder
- **data_analysis.ipynb**: Statistical and exploratory analysis, feature extraction, and decision tree modeling 📊🌳.
- **processed_data0.csv / processed_data1.csv / processed_data2.csv**: Segmented datasets with entity counts 🔢.
- **top_10_df_0_sorted.csv / top_10_df_1_sorted.csv / top_10_df_2_sorted.csv**: Top 10 entities per class 🏆.

### 3. Entity Extraction Folder
- **extract_entity.ipynb**: Extracts relevant entities using custom vocabularies 🔍.

### 4. Training and Classification Folder
- **training_process.ipynb**: Trains and evaluates classification models 🤖.

### 5. Bag of Words Folder
- **main.py**: Bag-of-words implementation using OOP 👜.
- **preprocess.py / vectorizer.py**: Preprocessing and vectorization scripts 🧹.
- **result_dataframe.csv / result_dataframe_v2.csv**: Output results 📄.

---

This project lays a solid foundation for future research, providing valuable insights and robust methodologies for addressing the challenges of land use and land cover changes in West Africa 🌍🚀.
