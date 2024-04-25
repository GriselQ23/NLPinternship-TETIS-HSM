import re
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_into_segments(text):
    # Split text into segments based on sentence boundaries (periods followed by whitespace)
    segments = re.split(r'\.+\s', text)
    return segments

def save_segments_with_word(segments, word_list, output_file, highlight_char='[', end_highlight_char=']'):
    with open(output_file, 'w') as file:
        for segment in segments:
            for word in word_list:
                if word.lower() in segment.lower():
                    # Highlight the word in the segment
                    highlighted_segment = segment.replace(word, f"{highlight_char}{word}{end_highlight_char}")
                    file.write(highlighted_segment + "\n")
                    break  # Once the segment is saved, move to the next segment
            else:
                continue  # Continue to the next segment if the current one doesn't contain any of the words


def main(pdf_file, word_list, output_file):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file)

    # Split text into segments
    segments = split_into_segments(text)

    # Save segments containing any of the words from the word list
    save_segments_with_word(segments, word_list, output_file, highlight_char='[', end_highlight_char=']')

pdf_file = '/home/grisel/Documents/internship/venv/corpus_english/sustainability-09-01917-v2.pdf'
word_list = ['crop', 'water', 'deforestation', 'savanna','fallow', 'urban', 'desertification', 'savannah', 'cultivation', 'grassland', 'agroforestry', 'erosion', 'maize', 'city', 'fruit', 'cotton', 'culture','urbanization','rice','shrubland', 'sorghum', 'reforestation', 'cassava', 'cashew', 'afforestation', 'corn', 'savannization', 'savane', 'meadows', 'urbanisation', 'cocoyam', 'roads', 'shrubs', 'citrus', 'paddy', 'cereals', 'vegetables', 'mustard', 'barley', 'sesame',	'manioc', 'taro', 'groundnuts', 'onions', 'beans', 'rye', 'asparagus', 'grasses', 'sunflower', 'squash', 'pawpaw', 'nuts', 'millets',	'pumpkin', 'peas']
output_file = 'segments_with_words.txt'  # Output file name
main(pdf_file, word_list, output_file)

