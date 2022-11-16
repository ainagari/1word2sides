''' This script was used to extract data from Semcor for experiments described in Section 2.3 and Appendix B of the paper.'''

import os
import bs4
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
#from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
from scipy.spatial.distance import jensenshannon
from copy import deepcopy
import pandas as pd
from itertools import combinations, chain
import pdb
import sys

random.seed(0)

sys.setrecursionlimit(100000)

sw = ENGLISH_STOP_WORDS #stop_words.ENGLISH_STOP_WORDS
sw = set(sw)
sw.add("have")
sw.add("do")
sw.add("can")
sw.add("could")
sw.add("should")
sw.add("would")
sw.add("must")
sw.add("shall")
sw.add("will")
sw.add("had")
sw.add("did")
sw.add("done")


def contains_digit(word):
    digits = '0123456789'
    for dig in digits:
        if dig in word:
            return True
    return False


def posmap(pos):
    if pos.startswith("N"):
        return 'n'
    if pos.startswith("V"):
        return 'v'
    if pos.startswith("J"):
        return 'a'
    if pos.startswith("R"):
        return 'r'


def lemma_meets_change_criterion(lemma_instances):
    '''The function outputs a dictionary with key(s) "senses" (previously: also supersenses)"
     containing the (super)sense partition chosen.'''
    # inspired from https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    senses_and_freq = dict()
    output = {"senses": None}
    for ins in lemma_instances:
        if ins['lexsn'] not in senses_and_freq:
            senses_and_freq[ins['lexsn']] = 0
        senses_and_freq[ins['lexsn']] += 1

    senses = list(senses_and_freq.keys())
    # if it is too computationally expensive, skip this lemma
    if len(senses) >= 25 or len(senses) <= 1:
        return output
    different_sized_combinations = chain(*[list(combinations(senses, i)) for i in range(1, len(senses))])
    pairs = list(tuple(sorted([x, tuple(set(senses) - set(x))])) for x in different_sized_combinations)
    pairs = list(set(pairs))
    random.shuffle(pairs)
    # Now check, for every combination, whether we can get up to 25 sentences. Stop when you find one.

    for pair in pairs:
        combi1, combi2 = pair
        sentences1 = sum([senses_and_freq[sense] for sense in combi1])
        sentences2 = sum([senses_and_freq[sense] for sense in combi2])
        if sentences1 >= min_sentences // 2 and sentences2 >= min_sentences // 2:
            output['senses'] = pair
            break

    return output


def read_corpus():
    sentences_by_lemma = dict()
    instances_by_lemma = dict()
    number_of_senses_by_lemma = dict()
    soup = bs4.BeautifulSoup(semcor.raw(), 'lxml')
    sentences = soup.find_all("s")
    print("loaded sentences")
    sentnum = 0
    for sentence in sentences:
        sentnum += 1
        if sentnum % 1500 == 0:
            print(sentnum)
        sentence = sentence.find_all(True, recursive=False)
        sentence_words = [str(t.string) for t in sentence if t.string != "\n"]
        for i, token in enumerate(sentence):
            if "lemma" not in token.attrs or "pos" not in token.attrs or "_" in token.string or "_" in \
                    token.attrs['lemma'] or token.attrs['lemma'] in sw:
                continue
            if not posmap(token['pos']):
                continue
            if contains_digit(token.string):
                continue
            lemmapos = (token['lemma'], posmap(token['pos']))
            if lemmapos not in number_of_senses_by_lemma:
                number_of_senses_by_lemma[lemmapos] = len(wn.synsets(token['lemma'], posmap(token['pos'])))
            if ";" in token['lexsn']:  # this means it has been assigned two senses. Omit these instances
                continue
            instance_to_add = dict()
            instance_to_add['sentence_words'] = sentence_words
            instance_to_add['sentence'] = " ".join(sentence_words)
            instance_to_add['position'] = i
            instance_to_add['word'] = str(token.string)
            if "lexsn" in token.attrs:
                instance_to_add['lexsn'] = token['lexsn']
            if "wnsn" in token.attrs:
                instance_to_add['wnsn'] = token['wnsn']

            if lemmapos not in sentences_by_lemma:
                sentences_by_lemma[lemmapos] = set()
                instances_by_lemma[lemmapos] = []
            if instance_to_add['sentence'] not in sentences_by_lemma[lemmapos]:
                instances_by_lemma[lemmapos].append(instance_to_add)
                sentences_by_lemma[lemmapos].add(instance_to_add['sentence'])
    return instances_by_lemma, sentences_by_lemma


def data_splitting(lemma_data, attested_senses=[], partition=[], mode="random"):
    num_sentences = len(kept_data[lp])
    parts = dict()
    sensekey = "lexsn"
    if mode == "random":
        parts[1] = lemma_data[:num_sentences // 2]
        parts[2] = lemma_data[num_sentences // 2:]
    elif mode == "change":
        # we have the partition. collect sentences until you have enough
        parts[1] = []
        parts[2] = []
        senses1, senses2 = partition
        for instance in lemma_data:
            sense = instance[sensekey]
            if sense not in senses1 and sense not in senses2:
                pdb.set_trace() # check if this is the case
            if sense in senses1 and len(parts[1]) < min_sentences // 2:
                parts[1].append(instance)
            elif sense in senses2 and len(parts[2]) < min_sentences // 2:
                parts[2].append(instance)
        if len(parts[1]) < min_sentences // 2 or len(parts[2]) < min_sentences // 2:
            print("Problem! Reached the end of the list!")
            pdb.set_trace()
    return parts


def assign_lemmas_to_modes(lemmas_with_enough_sentences, change_lemma_partitions):
    '''Assign lemmas to one of two modes: 'random' (if sentences are selected randomly) or 'change' if
    we want to maximize JSD for them (see Appendix B)'''
    lemmas_by_mode = dict()
    all_lemmas = lemmas_with_enough_sentences
    num_lemmas = len(all_lemmas)
    num_by_mode = {"change": num_lemmas // 6}
    lemmas_apt_for_change = list(change_lemma_partitions.keys())
    if len(lemmas_apt_for_change) < num_by_mode['change']:  # if there are not enough lemmas that can be assigned to the 'change' mode:
        print("Not enough lemmas can be assigned to the 'change' mode - reduce the desired quantity")
        pdb.set_trace()
    elif len(lemmas_apt_for_change) >= num_by_mode['change']:  # if instead we have enough such lemmas (1/3 of the total amount of lemmas):
        # randomly assign apt lemmas to this mode
        random.shuffle(lemmas_apt_for_change)
        lemmas_by_mode["change"] = lemmas_apt_for_change[:num_by_mode['change']]
        other_lemmas = [l for l in all_lemmas if l not in lemmas_by_mode["change"]]
        lemmas_by_mode["random"] = other_lemmas
    return lemmas_by_mode


def create_sense_vector(senses, sentences):
    sense_vector = np.zeros(len(senses), dtype=np.int8)
    sense_to_idx = dict(zip(senses, range(len(senses))))
    sensekey = "lexsn"
    for sentence in sentences:
        sense = sentence[sensekey]
        idx = sense_to_idx[sense]
        sense_vector[idx] += 1
    return sense_vector


def calculate_JSD(parts, X, global_lemma_info, attested_senses):
    subpart1 = parts[1][:X]
    subpart2 = parts[2][:X]
    k = "s_"
    sense_vector1 = create_sense_vector(attested_senses, subpart1)
    sense_vector2 = create_sense_vector(attested_senses, subpart2)
    global_lemma_info[k + "T1_X" + str(X)] = list(sense_vector1)  # 25 on each group
    global_lemma_info[k + "T2_X" + str(X)] = list(sense_vector2)
    global_lemma_info[k + "JSD_X" + str(X)] = jensenshannon(sense_vector1, sense_vector2, 2)  # base 2 as in schlechtweg et al
    return global_lemma_info


def prepare_instances_for_saving(parts):
    instances = []
    for partnum in parts:
        for i, sent in enumerate(parts[partnum]):
            instance = deepcopy(sent)
            instance["X25"] = 0  # an idx indicating to which part it belongs under X25. 1 for part 1, 2 for part 2, 0 for no part.
            for X in [3, 5, 10, 20, 25]:
                if i < X:
                    instance["X" + str(X)] = 1 if partnum == 1 else 2
                else:
                    instance["X" + str(X)] = 0
            instance['index'] = i if partnum == 1 else len(parts[1]) + i
            instances.append(instance)
    return instances


def save_individual_lemma(instances_for_saving, lemmapos, dirname):
    lpstr = "_".join(lemmapos)
    if not os.path.exists(dirname + lpstr):
        os.makedirs(dirname + lpstr)
    texts = [s['sentence'] for s in instances_for_saving]
    df = pd.DataFrame(instances_for_saving)  # put the data into a dataframe
    # delete the text-related variables (because we are saving the sentence separately)
    del df['sentence']
    del df['sentence_words']
    df.to_csv(dirname + lpstr + "/" + lpstr + "_info.csv", sep="\t", index=False)
    with open(dirname + lpstr + "/" + lpstr + "_sentences.txt", 'w') as out:
        for s in texts:
            out.write(s + "\n")

if __name__ == "__main__":

    out_dir = "semcor_representations/"
    min_sentences = 50
    max_sentences = 50  # so there will be 25 per class
    instances_by_lemma, sentences_by_lemma = read_corpus()
    print("I read the corpus")
    lemmas_with_enough_sentences = [lp for lp in sentences_by_lemma if len(sentences_by_lemma[lp]) >= min_sentences]

    ##### For every lemma, check if it is apt for belonging to the "change" mode (= JSD will be maximized).  We will want to keep all sentences for those.
    change_lemma_partitions = dict()  # here are only the lemmas for which we found a partition
    for l in lemmas_with_enough_sentences:
        chosen_partition = lemma_meets_change_criterion(instances_by_lemma[l])
        if not None in chosen_partition.values(): # If I could find a partition
            change_lemma_partitions[l] = {'senses': chosen_partition['senses']}
    print("We checked what lemmas can be of mode 'change'")

    ### Assign lemmas to modes
    lemmas_by_mode = assign_lemmas_to_modes(lemmas_with_enough_sentences, change_lemma_partitions)
    print("Modes have been assigned")

    ### shuffle
    kept_data = dict()
    for l in lemmas_with_enough_sentences:
        random.shuffle(instances_by_lemma[l])
        if l not in lemmas_by_mode['change']:
            kept_data[l] = instances_by_lemma[l][:max_sentences]
        else:
            kept_data[l] = instances_by_lemma[l]

    instances_by_lemma = []

    ####### START CREATING THE DATA
    print("starting split")

    jsd = []
    individual_lemmas = dict()

    for lp in kept_data:
        print(lp)
        # Necessary for all lemmas: check what senses are found in each sentence
        attested_senses = set()
        for sent in kept_data[lp]:
            attested_senses.add(sent['lexsn'])

        mode = [k for k in lemmas_by_mode if lp in lemmas_by_mode[k]][0]

        # Prepare the dicts
        lemma_sense_info = dict()
        lemma_sense_info['lemma'], lemma_sense_info['pos'] = lp
        lemma_sense_info['mode'] = mode

        if mode == "random":
            # Split the sentences into two classes (randomly)
            parts = data_splitting(kept_data[lp], mode="random")
        elif mode == "change":
            # Split the sentences into two classes (forcing specific distributions according to the lemma mode)
            sense_partition = change_lemma_partitions[lp]['senses'] if mode == "change" else []
            parts = data_splitting(kept_data[lp], attested_senses, partition=sense_partition, mode=mode)

        # Calculate and store the resulting sense distributions and JSDs for different amounts of sentences.
        for X in [3, 5, 10, 20, 25]:
            calculate_JSD(parts, X, lemma_sense_info, attested_senses)

        ### Save sentences
        instances_for_saving = prepare_instances_for_saving(parts)
        save_individual_lemma(instances_for_saving, lemmapos=lp, dirname=out_dir) #### TODO 3/11 prob remove
        jsd = lemma_sense_info

    #### Now save the global info (JSDs)
    jsd_df = pd.DataFrame(jsd)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    jsd_df.to_csv(out_dir + "global_info.csv", sep="\t", index=None)

