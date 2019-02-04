from pathlib import Path
import pandas as pd
import os
from os import listdir
import string
import re
import random

"""
#### FUNCTION DEFINITIONS ####
"""

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

def splitter(row, n):
    words = strip_formatting(row).split()
    output = [words[start:start+n] for start in range(0, len(words), n)]
    return output

def split_data_frame_list(df,
                       target_column,
                      output_type=float):
    '''
    Accepts a column with multiple types and splits list variables to several rows.

    df: dataframe to split
    target_column: the column containing the values to split
    output_type: type of all outputs
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    '''
    row_accumulator = []
    def split_list_to_rows(row):
        split_row = row[target_column]
        if isinstance(split_row, list):
          for s in split_row:
              new_row = row.to_dict()
              new_row[target_column] = s
              row_accumulator.append(new_row)
          if split_row == []:
              new_row = row.to_dict()
              new_row[target_column] = None
              row_accumulator.append(new_row)
        else:
          new_row = row.to_dict()
          new_row[target_column] = split_row
          row_accumulator.append(new_row)

    df.apply(split_list_to_rows, axis=1)
    new_df = pd.DataFrame(row_accumulator)
    return new_df


"""
##### DATA PREPERATION #####
"""

# Import metadata as DF; drop files that are restricted (part of EEBO Phase II)
metadata = pd.read_csv('/Users/au564346/Documents/research/LINK/LINK-master/dat/VEP_expanded_drama_1700_v2_txt/metadata/VEP_Expanded_Drama_1700_Metadata.csv')
metadata = metadata[metadata.status != 'Restricted']

# Create DF of one play per line
plays = pd.DataFrame(columns=['text'])
path = '/Users/au564346/Documents/research/LINK/LINK-master/dat/VEP_expanded_drama_1700_v2_txt/texts/'
for filename in os.listdir(path):
    if not filename.startswith('.'):
        with open(os.path.join(path, filename)) as f:
            text = f.read()
            current_df = pd.DataFrame({'text': [text]})
            plays = plays.append(current_df, ignore_index=True)

# Join up plays with author and genre metadata
plays['genre'] = metadata["genre"].to_frame().reset_index(drop=True)
plays['author'] = metadata["author"].to_frame().reset_index(drop=True)
plays["year"] = metadata["date of writing"].to_frame().reset_index(drop=True).astype(int)

# Reset column index
columns = ["text", "genre", "author", "year"]
plays = plays.reindex(columns=columns)
# Split string into list of lists
plays['text'] = plays["text"].apply(lambda text: splitter(text, 10))
# Explode DataFrame and keep author/genre metada in place
corpus = split_data_frame_list(plays, target_column='text')
# List of lists back to string
corpus["text"] = corpus["text"].apply(lambda x: ' '.join(map(str, x)))

years = list(range(1610, 1631))
subset = plays.loc[plays.year.isin(years)].reset_index()
del subset["index"]

# fastText prep
percent_test_data = 0.2
training_data = Path("/Users/au564346/Desktop/fasttext_dataset_training.txt")
test_data = Path("/Users/au564346/Desktop/fasttext_dataset_test.txt")

with training_data.open("w") as train_output, test_data.open("w") as test_output:

    for i in range(len(corpus)):

        author = corpus['author'][i]
        author = author.replace(" ","")
        author = author.replace(",","")

        text = corpus['text'][i].replace("\n", " ")
        fasttext_line = "__label__{} {}".format(author, text)

        if random.random() <= percent_test_data:
            test_output.write(fasttext_line + "\n")
        else:
            train_output.write(fasttext_line + "\n")


"""
Now run fastText in terminal:
# train
fasttext supervised -input fasttext_dataset_training.txt -output EMD_model
# test
fasttext test EMD_model.bin fasttext_dataset_test.txt
# top 2
fasttext test EMD_model.bin fasttext_dataset_test.txt 2

# predict
fasttext predict reviews_model.bin -

Bigrams and trigrams to increase accuracy:
# bigrams_train
fasttext supervised -input fasttext_dataset_training.txt -output EMD_model_ngrams -wordNgrams 2
# bigram_test
fasttext test EMD_model_ngrams.bin fasttext_dataset_test.txt
# trigrams_train
fasttext supervised -input fasttext_dataset_training.txt -output EMD_model_3grams -wordNgrams 3
# trigram_test
fasttext test EMD_model_3grams.bin fasttext_dataset_test.txt
"""
