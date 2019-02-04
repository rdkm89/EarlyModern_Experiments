"""
Pre-processing script for Strucutural Topic Modelling
Cf stm.R
"""

print("importing packages...")
import os, re, string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words

print("defining functions..")
# Function definitions
def strip_formatting(df_row):
    words = df_row.str.replace('[{}]'.format(string.punctuation), ' ')
    lines = words.str.replace("\n", "")
    cleaned = lines.str.lower()
    return cleaned

def splitter(row, n):
    words = re.sub("[^\w]", " ",  row).split()
    output = [words[start:start+n] for start in range(0, len(words), n)]
    return output

def split_data_frame_list(df,target_column,output_type=float):
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

print("reading metadata...")
# Import metadata as DF; drop files that are restricted (part of EEBO Phase II)
metadata = pd.read_csv('/Users/au564346/Documents/research/LINK/data/LINK-master/dat/VEP_expanded_drama_1700_v2_txt/metadata/EM_Drama.Metadata.csv', sep=";")
metadata = metadata[metadata.status != 'Restricted']

print("reading plays...")
# Create DF of one play per line
plays = pd.DataFrame(columns=['documents'])
path = '/Users/au564346/Documents/research/LINK/data/drama_queens/Corpus without MA og anon'
for filename in os.listdir(path):
    if not filename.startswith('.'):
        with open(os.path.join(path, filename)) as f:
            text = f.read()
            current_df = pd.DataFrame({'documents': [text]})
            plays = plays.append(current_df, ignore_index=True)

print("joining plays + metadata...")
# Join up plays with author and genre metadata
plays['genre'] = metadata["genre"].to_frame().reset_index(drop=True)
plays['author'] = metadata["author"].to_frame().reset_index(drop=True)
plays['year'] = metadata["date of writing"].to_frame().reset_index(drop=True).astype(int)

print("removing uppercase and punctuation...")
# Clean strings using strip_formatting function
plays['documents'] = [cleaned for cleaned in strip_formatting(plays["documents"])]

print("removing user-defined stopwords...")
# User defined stopwords
# Create list of stopwords from various sources
stop_1 = get_stop_words("english")
stop_2 = stopwords.words("english")
#Combine as one list
stops = list(set(stop_1 + stop_2))
# Extend as you see fit!
stops.extend(['one', 'thou', 'thy', 'thee',                               # Pronouns
              'sir', 'lord', 'lords', 'lady', 'ladies',                   # Honorifics, etc
              'madam', 'master', 'men', 'women', 'mrs',
              'may', 'must', 'like', 'will', 'shall', 's', 'art', 'say',  # Modal verbs
              'yet', 'well', 'good',                                      # Adverbs, adjectives
              'let', 'come'])                                             # Imperatives
# Stopwords
plays['documents'] = plays['documents'].apply(lambda x: " ".join([item for item in (str(x)).split(" ") if item not in stops]))

#print("stemming docs...")
# SnowballStemmer
#sb = SnowballStemmer('english')
# Stemming
#plays["documents"] =  plays["documents"].apply(lambda x: " ".join([sb.stem(word) for word in (str(x)).split(" ")]))

print("splitting plays into equal chunks...")
# Split string into list of lists using splitter function
plays['documents'] = plays['documents'].apply(lambda x: splitter(x, 1000))
# Explode DataFrame and keep author/genre metada in place
corpus = split_data_frame_list(plays, target_column='documents')
# List of lists back to string
corpus['documents'] = corpus['documents'].apply(lambda x: ' '.join(map(str, x)))

print("saving output...")
# 1-based indexing for R; Save to csv
corpus.index += 1
corpus.to_csv("/Users/au564346/Desktop/corpus.csv", header=True)
