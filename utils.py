
import os
import csv
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from string import punctuation
import codecs
import numpy as np
import pandas as pd


punctuation = set(punctuation)
numbers = set('1234567890')
sws = stopwords.words('english')
sws.extend(["ca","okay","yeah","gon","do","um","yes","get","uh","would","something","rt"])
sws = set(sws)


def remove_repeated_sentences_IBM(data):
    '''
    remove sentences that are repeated in the IBM data *with the same target*
    Args:
        data: list of dicts
    Returns: new_data, list of dicts containing unique sentences
    '''
    sentences = dict()
    new_data = []
    for ins in data:
        sentence, target = ins['Sentence'], ins["Target"]
        if sentence in sentences and target in sentences[sentence]:
            continue
        else:
            new_data.append(ins)
            if sentence not in sentences:
                sentences[sentence] = set()
            sentences[sentence].add(target)
    return new_data



def trim_sentence(sentence_words, target_keywords, target):
    '''
    Args:
        sentence_words: list of strs (words)
        target_keywords: most important words in the target this sentence corresponds to
        target: the original target
    Returns:
        sentence_words: list of strs (words)
        bool, if True, it means we detected some overlap but the sentence was left unchanged

    '''
    # see if the first X words are the target or the target + not/nt
    kws = len(target_keywords)
    len_target = len(target.split())
    targetset = set(target.lower().split())
    targetset.add("not")
    targetset.add("n't")

    if sentence_words[:len_target] == target.lower().split() or set(sentence_words[:len_target+1]).issubset(targetset):
        # check if the next word is because, as, since, comma or stop.

        if "because" in sentence_words[len_target:len_target+3]:
            position = sentence_words.index("because")
            if sentence_words[position + 1] == "of":
                position += 1
        elif "as" in sentence_words[len_target:len_target+3]:
            position = sentence_words.index("as")
        elif "since" in sentence_words[len_target:len_target + 3]:
            position = sentence_words.index("since")
        elif "," in sentence_words[len_target:len_target + 3]:
            position = sentence_words.index(",")
        elif "." in sentence_words[len_target:len_target + 3]:
            position = sentence_words.index(".")
        else:
            q=2 #to, so, so that, due to, for, in ordre to
            return sentence_words, True # True means it has overlap but it didnt change
        return sentence_words[position + 1:], False

    else:
        return sentence_words, False




def determine_target_keywords(data):
    targets = set([ins['Original Target'] for ins in data])
    target_keywords = dict()
    # I basically postag the target and keep only nouns, verbs and adjectives
    for target in targets:
        postagged_target = pos_tag(word_tokenize(target.lower()), tagset="universal")
        target_keywords[target] = set([w for w, p in postagged_target if p in ["NOUN","VERB","ADJ"] and w not in sws])
    return target_keywords



def read_IBM_dataset(dataset_name, trim=True, path='Data/', return_df=False):
    '''
    Reads in an IBM dataset (for example ArgQP (30k))
    Args:
        dataset_name: string: 30k (ArgQP in the paper), CS, ArgKP, XArgMining-Arg, XArgMining-Evi
        return_df: whether we want to return a dataframe
        trim: bool, if True, we shorten some sentences that have a high overlap with the target
        (for example, many sentences start like "We should ban abortion because ..." - we only keep what's after "because")

    Returns: data, either a list of dicts (each dict corresponds to one sentence) or, if return_df, a dataframe
    '''

    if dataset_name == "30k": # ArgQP
        print('here')
        fn = "IBM_Debater_(R)_arg_quality_rank_30k/arg_quality_rank_30k.csv"
        cols_to_del = []
        cols_to_change = [('topic', "Target"), ('argument', "Sentence"), ('stance_WA', "Stance")]
        replacement_dict = {1: "FAVOR", -1: "AGAINST", 0: "NONE"}

    df = pd.read_csv(path + fn, sep=",")
    for col in cols_to_del:
        del df[col]
    for oldcol, newcol in cols_to_change:
        df[newcol] = df[oldcol]
        del df[oldcol]
    df["Stance"].replace(replacement_dict, inplace=True)

    if return_df:
        print("incompatible with trimming")
        return df

    print("REMOVING instances with low stance clarity")
    if dataset_name == "30k":
        clear = [r for i, r in df.iterrows() if r['stance_WA_conf'] >= 0.6]
        df = pd.DataFrame(clear)

    data = df.to_dict('records')

    for ins in data:
        tokens = word_tokenize(ins['Sentence'])
        ins["tokens"] = [t.lower() for t in tokens]  # Reminder: this is lowercased, tokenized_tweet is not
        ins["Original Target"] = ins["Target"]
        ins["Target"] = ins["Target"] + "%" + dataset_name
    data = remove_repeated_sentences_IBM(data)

    if trim:
        target_keywords = determine_target_keywords(data)
        for ins in data:
            target = ins["Original Target"]
            ins["tokens"], it_has_overlap_but_unchanged = trim_sentence(ins["tokens"], target_keywords[ins["Original Target"]], target)

    return data




def read_twitter_dataset(dataset_name, debugging=False, path='Data/'):
    '''used to be 'read_dataset'.
    dataset_name can be "semeval2016", "covid19", "pstance"
    if debugging=True, we only process and return info for 2,000 tweets
    :return: data (a list of tweets, each tweet is a dictionary) '''
    data = []
    if not path:
        diri = "Data/"
    elif path:
        diri = path
    if dataset_name == "semeval2016":
        diri += "semeval2016_stance/StanceDataset/"
    elif dataset_name == "pstance":
            diri += "PStance/"
    elif dataset_name == "covid19":
        diri += "covid_unabridged_dataset/"
    print("omitting 'noisy' versions of the datasets")
    for fn in os.listdir(diri):
        if fn.endswith(".csv") and not "noisy" in fn:
            with codecs.open(diri + fn, 'r', encoding='utf-8',
                             errors='ignore') as f:
                csvreader = csv.DictReader(f, delimiter=",")
                data.extend(list(csvreader))

    if debugging:
        print("DEBUGGING MODE!")
        data = data[:2000]

    # Update the names of the targets so they include the dataset name too,
    # because one target (Donald Trump) is present in two datasets
    tt = TweetTokenizer()
    for ins in data:
        ins['Target'] = ins['Target'].replace(" ","_")
        ins['Target'] += "%" + dataset_name
        tokens = tt.tokenize(ins['Tweet'])
        ins["tokenized_tweet"] = tokens # not lowercased, original but tokenized tweet
        ins["tokens"] = [t.lower() for t in tokens if t.lower() != "#semst" and "\n" not in t.lower()] # this is lowercased and cleaned
    return data


# FUNCTIONS FOR LOADING A TXT FILE WITH VECTORS
# First two functions taken/adapted from https://github.com/NLPrinceton/ALaCarte/blob/master/alacarte.py
def make_printable(string):
    '''returns printable version of given string
    '''
    return ''.join(filter(str.isprintable, string))

def load_vectors(vectorfile):
    '''loads word embeddings from .txt
    Args:
      vectorfile: .txt file in "word float ... " format
    Returns:
      (word, vector) generator
    '''

    SPACE = ' '
    FLOAT = np.float32
    words = set()
    with open(vectorfile, 'r') as f:
        for line in f:
            index = line.index(SPACE)
            word = make_printable(line[:index])
            if not word in words:
                words.add(word)
                yield word, np.fromstring(line[index + 1:], dtype=FLOAT, sep=SPACE)

# Function taken/adapted from https://github.com/NLPrinceton/ALaCarte/blob/master/alacarte.py
def dump_vectors(vector_dict, vectorfile):
  '''dumps embeddings to .txt
  Args:
    generator: (gram, vector) generator; vector can also be a scalar
    vectorfile: .txt file
  Returns:
    None
  '''

  with open(vectorfile, 'w') as f:
    for gram, vector in vector_dict.items():
      numstr = ' '.join(map(str, vector.tolist())) if vector.shape else str(vector)
      f.write(gram + ' ' + numstr + '\n')



def find_semcor_paths(semcor_dir="semcor_representations/"):
    dirs = [semcor_dir + diri for diri in os.listdir(semcor_dir) if os.path.isdir(semcor_dir + diri)]
    return dirs


