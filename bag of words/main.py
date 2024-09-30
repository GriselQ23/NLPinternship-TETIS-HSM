from preprocess import Preprocess
from vectorizer import Vectorizer 
import pandas as pd 
import os

def main(folder_path):
    # Initialize an empty DataFrame to store TF-IDF scores
    result_df = pd.DataFrame()

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            # Construct the full path to the PDF file
            pdf_path = os.path.join(folder_path, filename)

            # Preprocess the text
            data_prep = Preprocess(pdf_path)
            prepared_text = data_prep.preprocess()

            # Compute TF-IDF scores
            nomenclature = '/home/grisel/Documents/internship/venv/scrips/nomenclature_v2.csv'
            tf_idf_score = Vectorizer(prepared_text, nomenclature)
            tf_idf_df = tf_idf_score.computation()

            # Add document name as a column in the DataFrame
            tf_idf_df['Document'] = filename

            # Append the TF-IDF scores DataFrame to the result DataFrame
            result_df = pd.concat([result_df, tf_idf_df], ignore_index=True)

    return result_df


if __name__ == "__main__":
    folder_path = '/home/grisel/Documents/internship/venv/corpus_english'
    result_dataframe = main(folder_path)
    rows, columns = result_dataframe.shape
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {columns}")

    # Save DataFrame to CSV file
    output_csv_path = '/home/grisel/Documents/internship/venv/scrips/preprocess/result_dataframe_v2.csv' 
    #Ensure that the csv will be downloaded as a csv file
    result_dataframe.to_csv(output_csv_path, index=False)
    print(f"DataFrame saved to {output_csv_path}")