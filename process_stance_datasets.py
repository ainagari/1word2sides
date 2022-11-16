'''
With this script we preprocess the stance datasets splitting the data for every target into sentence sets
'''

import argparse
from utils import *
from random import shuffle
import random
from nltk import pos_tag, WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from copy import copy
import json


sws.add("#semst")

random.seed(9)

def find_common_words_stance(preprocessed_data, min_freq_per_subset=3, min_utterance_length=5):
    '''
        preprocessed_data: list of dicts, each dict is a sentence that has been processed
        min_freq_per_subset: int, minimum number of times a word must appear at each side of a comparison for us to consider it
        min_utterance_length: int, minimum number of tokens that a sentence should have for us to consider it
    Returns: data_per_subset_wordinfo, dict where keys are the name of a subset (e.g. FAVOR-1).
    values are dicts, where keys are lemma_pos. and the value is a list of sentences in that subset containing that lemma_pos.
    '''
    # List of the words that a priori could be included for every subset, and their frequency
    lemmapos_freq = dict()
    for subset in preprocessed_data:
        words_uttered = []
        for utt in preprocessed_data[subset]:
            # if utterance is too short, skip it (we don't include them nor count freqs from them)
            if len(utt['preprocessed_utterance']) < min_utterance_length:
                continue

            words_uttered.extend(list(set([l + "_" + p for w, l, p in utt[
                'preprocessed_utterance']])))  # (we don't want to count more than one instance per sentence -> set)
        lemmapos_freq[subset] = Counter(words_uttered)

    # look for words that are common in the subset pairs of interest
    data_per_subset_wordinfo = dict()
    kept_words = dict()
    for subset in preprocessed_data:
        if "NONE" in subset:
            continue
        kept_words[subset] = []
        for word in lemmapos_freq[subset]:
            if lemmapos_freq[subset][word] >= min_freq_per_subset:
                found = []
                for other_subset in preprocessed_data:
                    if other_subset != subset:
                        if lemmapos_freq[other_subset][word] >= min_freq_per_subset:
                            found.append(other_subset)
                if found:
                    kept_words[subset].append(word)

    for subset in kept_words:
        data_per_subset_wordinfo[subset] = dict()
        for utt in preprocessed_data[subset]:
            previous_lp = set()
            for w, l, p in utt["preprocessed_utterance"]:
                lp = l + "_" + p
                if lp in kept_words[subset] and lp not in previous_lp:
                    if lp not in data_per_subset_wordinfo[subset]:
                        data_per_subset_wordinfo[subset][lp] = []
                    data_per_subset_wordinfo[subset][lp].append(utt)
                    previous_lp.add(lp)

    return data_per_subset_wordinfo



def save_with_dialign_format(plain_convs, path):
	'''
	Args:
		plain_convs: a dict where keys are either (a) a conversation ID (in the case of dialogs) or (b) a comparison name corresponding to a target (in the case of stance)
		values: a list of tuples (subsetname, sentence string)
		path: directory where we want to save this (filenames are the conversationID/comparisonname)
	Returns: None, writes to file
	'''
	if not os.path.exists(path):
		os.makedirs(path)
	for conversation in plain_convs:
		with open(path + conversation + ".tsv",'w') as out:
			for speaker, utterance in plain_convs[conversation]:
				out.write(speaker + ":\t" + utterance + "\n")



def split_into_subsets(data, target):
    '''This creates the sets Pf, Qf (FAVOR1, FAVOR2...)'''
    data_by_subset = dict()
    stances = set([ins['Stance'] for ins in data])
    tweets_by_stance = {stance: [ins for ins in data if ins["Stance"] == stance and ins["Target"] == target]
                        for stance in stances}
    for stance in tweets_by_stance:
        shuffle(tweets_by_stance[stance])
    # FAVOR 1 (Pf), FAVOR 2 (Qf), AGAINST 1 (Pf), AGAINST 2 (Qf)
    for stance in tweets_by_stance.keys():
        data_by_subset[stance + '-1'] = tweets_by_stance[stance][: len(tweets_by_stance[stance]) // 2]
        data_by_subset[stance + '-2'] = tweets_by_stance[stance][len(tweets_by_stance[stance]) // 2:]

    return data_by_subset



def preprocess_texts(data, twitter):
    '''
    Loads a lemmatizer and calls process_utterance on every sentence, which lemmatizes them and finds potentially relevant lemmas (nouns and verbs)
        Args:
        data: list of dicts
        twitter: bool, if True, data comes from Twitter

    Returns:
        modifies data, including the newly extracted info
    '''
    lemmatizer = WordNetLemmatizer()
    for utterance in data:
        process_utterance(utterance, lemmatizer, twitter=twitter)
    return data

def twitter_aware_lemmatization(w, pos, lemmatizer, twitter=False):
	if twitter and w[0] in {"@", "#"}:
		l, pos = w, 'NOUN'  # we'll treat them all like nouns (mentions and hashtags)
	else:
		if pos in ["NOUN","VERB","ADJ"]:
			l = lemmatizer.lemmatize(w, pos[0].lower())
		elif pos == "ADV":
			l = lemmatizer.lemmatize(w, 'r')
		else:
			l = lemmatizer.lemmatize(w)
	return l, pos


def process_utterance(utterance, lemmatizer, twitter=True):
	'''
	Lemmatize an utterance and find the lemmas that are potentially relevant in it. Only nouns and verbs.
	Hashtags and mentions count as nouns.
	Args:
		utterance: dict with key "tokens" containing the tokens of a sentence
		lemmatizer: typically nltk's wordnetlemmatizer (taken care of in process_stance_for_alignment)
		twitter: bool, if True, the sentence is a tweet and we need to treat hashtags and mentions carefully

	Returns:
		nothing, it modifies utterance, adding "preprocessed_utterance", "lemma_positions", "lemmas"
	'''
	tokens = utterance['tokens']
	postagged = pos_tag(tokens, tagset='universal')
	lemma_positions = dict()
	important_lemmas = []
	all_lemmas = []
	for i, (w, pos) in enumerate(postagged):
		lemma, pos = twitter_aware_lemmatization(w, pos, lemmatizer, twitter=twitter)
		all_lemmas.append(lemma)
		keep = True
		if w not in sws and len(w) > 2: ## omit words shorter than 2 letters
			if twitter and (w[0] in {"@", "#"}):
				keep = True
			else:
				if pos in ["NOUN", "VERB"]:
					for char in w:
						if char in punctuation or char in numbers:
							keep = False
							break
				elif pos not in ["NOUN", "VERB"]:
					keep = False
			if keep:
				important_lemmas.append((w, lemma, pos))
				if lemma + "_" + pos not in lemma_positions:
					lemma_positions[lemma + "_" + pos] = []
				lemma_positions[lemma + "_" + pos].append(i)
	utterance["preprocessed_utterance"] = important_lemmas
	utterance["lemma_positions"] = lemma_positions
	utterance["lemmas"] = all_lemmas


def save_subset_data_stance(data_by_subset_wordinfo, path):
    '''
    Save the preprocessed, split data
    Args:
        data_by_subset_wordinfo: the output of find_common_words_stance
        path: where to save it
    Returns: None, writes to file
    '''
    for subset in data_by_subset_wordinfo:
        # throw an error if dir exists
        pathsubset = path + str(subset) + "/"
        os.mkdir(pathsubset)
        for lp in data_by_subset_wordinfo[subset]:
            pathlemma = pathsubset + lp + "/"
            if not os.path.exists(pathlemma):
                os.makedirs(pathlemma)
            sentences = []
            infos = []
            try:
                for utt in data_by_subset_wordinfo[subset][lp]:
                    sentences.append(" ".join(utt['tokens']))
                    uttcopy = copy(utt)
                    del uttcopy['tokens']
                    infos.append(uttcopy)
            except python-BaseException:
                q = 2
            if len(sentences) != len(infos):
                q = 2

            with open(pathlemma + "sentences.txt", 'w') as out:
                out.write("\n".join(sentences))
            infosjson = json.dumps(infos)
            with open(pathlemma + "info.txt", 'w') as out:
                out.write(infosjson)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="semeval2016", help="semeval2016, covid19, pstance, 30k")
    args = parser.parse_args()

    # Load dataset
    if args.dataset_name in ["covid19","pstance","semeval2016"]:
        data = read_twitter_dataset(args.dataset_name, path="Data/")
        twitter = True
    elif args.dataset_name in ["30k"]:
        data = read_IBM_dataset(args.dataset_name, trim=True, path='Data/')
        twitter = False

    # First, preprocess the data (lemmatize, postag...)
    data = preprocess_texts(data, twitter=twitter)

    # Split the data for each target into different sets and create the comparisons
    targets = list(set([x['Target'] for x in data]))
    for target in targets:
        data_by_subset = split_into_subsets(data, target)  # We randomly split data for each stance into two halves
        data_by_subset_wordinfo = find_common_words_stance(data_by_subset) # we look for words that appear a minimum number of times in more than one subset

        # Store the sentences in each comparison
        plain_convs = dict()
        convs = [('same-stance_FAVOR','FAVOR-1','FAVOR-2'),('same-stance_AGAINST', 'AGAINST-1','AGAINST-2'),
                 ('different-stance_FAVOR-AGAINST(1)', 'FAVOR-1','AGAINST-2'),('different-stance_FAVOR-AGAINST(2)', 'FAVOR-2','AGAINST-1')]
        for convname, subset1, subset2 in convs:
            plain_convs[convname] = []
            for subset in (subset1, subset2):
                for utterance in data_by_subset[subset]:
                    plain_convs[convname].append((subset, " ".join(utterance['tokens'])))

        # Save sentences in a dialign-compatible format
        save_with_dialign_format(plain_convs, path="dialign_data/" + args.dataset_name + "/" + target + "/")

        # Save the comparisons
        out_dir = "stance_split_data/" + args.dataset_name + "/" + target + "/"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_subset_data_stance(data_by_subset_wordinfo, path=out_dir)



