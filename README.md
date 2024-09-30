# NLPinternship-TETIS-HSM
1. Preprocess: in this folder you have  the corpus and dataset where is create_dataset.ipynb the input is a tei file this was manually converted with: https://huggingface.co/spaces/lfoppiano/grobid also there is a folder with some tei documents from the corpus. Also is the nomenclature. The output is the dataset in a csv format
2. training and classification 
Here is the script for and the models I use for classify at the end  
3. Entity extraction
This is the code to extract the entities we are interested, there is a csv file with LCprocess and LCvocabulary you need both.  
4. Data Analysis 
As entry you have processes_data0.csv, processes_data1.csv, and processes_data2.csv, this are segments with the sum of all entities extracted, there are some you don't need like ORG but you can filter. 
In the file data_analysis there is an analysis like find the most ocurrance, there is a top 10,
also and analysis per class, boxplot, correlations and finally a decision tree to find the most important features with the variable target class. 
In the files top_10_df_0_sorted.csv  top_10_df_1_sorted.csv  top_10_df_2_sorted.csv there is the top 10 of each class.  
5. Bonus: Bag of words
This is an implementation for only bag of words, but has an approach of POO and its very well implemented. you have to run only main.py 
