'''
This script reads in all 4 stance datasets considered in the paper
and it calculates the tf idf PER TARGET (one target = one document)
and it saves this info in the dialign_data/ directory
'''


from utils import read_twitter_dataset, read_IBM_dataset
from process_stance_datasets import preprocess_texts
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import argparse


def calculate_tfidf(data, targets):
    # Instantiate vectorizer. Adapted for tweets
    vectorizer = TfidfVectorizer(max_df=len(targets)-2, token_pattern=r'\b\w\w+\b|(?<!\w)@\w+|(?<!\w)#\w+') # (Reminder that one target from SemEval2016 will not be used at all)
    # One target = one document
    by_document = []
    for target in targets:
        text = ""
        tweets_in_doc = [t for t in data if t["Target"] == target]
        for tid in tweets_in_doc:
            text += " ".join(tid["lemmas"]) + " "
        by_document.append(text)

    # tfidf should be a targets x terms matrix (document-term matrix) showing the importance of each term in each document.
    tfidf = vectorizer.fit_transform(by_document).toarray()

    return vectorizer, tfidf


def save_tfidf_data(vectorizer, tfidf, targets, target, path):
    idx_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
    targetnum = targets.index(target)
    word_tfidf = dict()
    # make a {word: tfidf} dict
    for idx, value in enumerate(tfidf[targetnum]):
        word = idx_to_word[idx]
        word_tfidf[word] = value
    # sort it from highest to lowest tfidf
    sorted_word_tfidf = sorted(word_tfidf.items(), key=lambda k:k[1], reverse=True)
    # only print them if they are not 0
    sorted_word_tfidf_nonzero = [(k,v) for k,v in sorted_word_tfidf if v > 0]
    out_fn = path + "tfidf.tsv"
    with open(out_fn, 'w') as out:
        for w, t in sorted_word_tfidf_nonzero:
            out.write(w+"\t"+str(t)+"\n")




if __name__ == "__main__":

    all_data = []
    for dataset_name in ["semeval2016","30k"]: #["covid19","pstance","semeval2016","30k"]:
        if dataset_name in ["covid19","pstance","semeval2016"]:
            data = read_twitter_dataset(dataset_name, path="Data/")
            twitter = True
            # Not interested in sentences with stance "NONE"
            data = [x for x in data if x['Stance'] in ['FAVOR','AGAINST']]
        elif dataset_name in ["30k"]:
            data = read_IBM_dataset(dataset_name, trim=True, path="Data/")
            twitter = False

        # Preprocess
        data = preprocess_texts(data, twitter=twitter)
        all_data.extend(data)

    targets = list(set([x['Target'] for x in all_data]))


    # Calculate tf idf of all datasets at once
    vectorizer, tfidf = calculate_tfidf(all_data, targets)

    # Save
    for target in targets:
        dataset_name = target.split("%")[-1]
        out_dir = "dialign_data/" + dataset_name + "/" + target + "/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_tfidf_data(vectorizer, tfidf, targets, target, path=out_dir)


