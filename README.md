
# One Word, Two Sides ⚖️

This repository contains data and code for the papers:

Aina Garí Soler, Matthieu Labeau and Chloé Clavel (2022). [One Word, Two Sides: Traces of Stance in Contextualized Word Representations](https://aclanthology.org/2022.coling-1.347/). Proceedings of the 29th International Conference on Computational Linguistics (COLING), Gyeongju, Korea, October 12-17

Aina Garí Soler, Matthieu Labeau and Chloé Clavel (2023). Un mot, deux facettes : traces des opinions dans les représentations contextualisées des mots. Actes de la 30e Conférence sur le Traitement Automatique des Langues Naturelles (TALN), Paris, France, June 5-9


(Disclaimer: please forgive redundancy, at earlier stages of the project it made sense to organize information a certain way, later on I built upon the already-written code and didn't have time to focus on improving that :))

## Data

### Semcor Data

* The directory in ``semcor_representations.zip`` contains:
	- a ``global_info.csv`` file with the Jensen Shannon Divergence (JSD) information. For each lemma, we have the JSD for each data size X (e.g. ``s_JSD_X3`` is the Jensen Shannon Divergence of a lemma when using three sentences per side). ``s_T1_X3`` and ``s_T2_X3`` are the sense distributions that were used to calculate ``s_JSD_X3``. ``T1`` and ``T2`` are the sentence sets (_P_ and _Q_). 
	- the data extracted from Semcor (described in Section 2.3 and Appendix B) organized by lemma. Inside of each lemma folder there are two files. For example, for the lemma *accept_v*, we have:

		- ``accept_v_sentences.csv``: extracted sentences containing this lemma, one per line. They are already tokenized (.split() is enough).
		- ``accept_v_info.csv``: This file contains additional information about each sentence in the previous file. ``position`` is the index of the target word in the sentence (respecting its original tokenization). ``lexsn``, ``wnsn`` and ``supersense`` are sense information from WordNet (JSD was calculated based on lexsn). Values in columns ``X25``, ``X3``... indicate whether, in this X-sized subset, the instance belongs to set *P* (1), *Q* (2) or whether it is excluded (0). ``index`` indicates the line in the ``_sentences.csv`` file that an instance corresponds to, starting at line 0.
	
When extracting vector representations for Semcor data, these will be saved in this directory (more information below).


* The directory ``semcor_predictions`` contains the similarity predictions made by all the embeddings tested on Semcor data (Sections 2.3, 2.4). Each folder corresponds to a representation type, and contains a ``cosine.csv`` file. This file has the cosine similarities for every lemma and each X.


### Stance Data

* The stance datasets should be placed, once downloaded, in the ``Data`` folder.

* The directory inside of ``stance_split_data.zip`` will contain the datasets already split by sentence sets. Here we only include an example (``stance_alignment_data/30k/Homeschooling should be banned%30k/``), but the directory can be populated using the ``process_stance_for_alignment.py`` script (see [Code](##Code) section). There will be one folder per stance dataset (``30k`` refers to the BM-ArgQ-Rank-30kArgs dataset, ArgQ in the paper). Inside a stance dataset folder, each directory will correspond to a different target in this dataset (e.g. ``face_masks%covid19``). Inside, we find the *P_f*, *P_a*, *Q_f* and *Q_a* sentence sets, called, respectively, ``FAVOR-1``, ``AGAINST-1``, ``FAVOR-2``, ``AGAINST-2``. They contain the sentences organized by lemma. Inside a lemma folder (e.g. ``agree_VERB``) we will find two files:
	- ``sentences.txt``: containing one sentence per line, already tokenized (.split() is enough).
	- ``info.txt`` containing additional information about each sentence in json format. Each element corresponds to one sentence.



## Code


### Semcor experiments

#### How to extract the Semcor data
* Use the ``create_semcor_data.py`` script. You will need to download the semcor corpus through ``nltk``. By default, the data will be saved in the directory ``semcor_representations``. 

#### Obtaining representations for the Semcor data

* ``extract_semcor_representations.py`` to obtain context2vec or BERT representations (``--vector_type c2v/bert``) for semcor data. Representations will be saved in the same directory where the dataset is found (the unzipped ``semcor_representations/``). For context2vec, you should download the contents of the [context2vec repository](https://github.com/orenmel/context2vec) and place them in a folder called ``context2vec``. You have to provide the path to context2vec model parameters with the argument ``c2v_filename``.
* To obtain À la Carte embeddings, you should place the contents of the [ALC repository](https://github.com/NLPrinceton/ALaCarte) in a folder called ``ALC``. Include in this folder our script ``ALC/get_alc_semcor.py``. You can obtain these embeddings from the ``ALC`` directory with the following command:
``python get_alc_semcor.py --source [path to GloVe embeddings] --matrix [path to transform matrix in ALC/transform/] --embeddings_name 840B.300d --lower --corpus_dir ../semcor_representations/ --create-new``.   

#### Making predictions and evaluating
* Use the script ``make_semcor_predictions.py`` to calculate cosine similarities in semcor with the embeddings calculated as described above. Use the argument ``vector_type`` to specify the embedding type (``bert``, ``c2v`` or ``ALC``). If choosing ``ALC``, provide the embedding size with ``--vector_size`` (e.g. ``840B.300d``). Results will be saved in the ``semcor_predictions`` directory.

* Plots of the results can be created with the notebook ``semcor_evaluation.ipynb``.

### Stance experiments

#### Splitting stance datasets into sentence sets

First, you must download the stance datasets and unzip them in the ``Data/`` directory.

* The script ``process_stance_datasets.py`` serves to split a stance dataset into sentence sets. Indicate the name of the dataset (``semeval2016, covid19, pstance, 30k``) with the ``--dataset_name`` argument (if needed, paths to the datasets can be modified in the ``utils.py`` script: ``read_IBM_dataset() / read_twitter_dataset()`` ). The data split into sentences will be saved in ``stance_split_data``. Sentences will be saved in a different format (used for tf-idf calculation) in ``dialign_data``.
* To calculate tf-idf information, use the script ``calculate_stance_tfidf.py`` after having run ``process_stance_datasets.py``. Again, you can indicate the name of the dataset with the ``--dataset_name`` argument. Tf-idf values will be saved in the ``dialign_data`` folder.


- ``extract_stance_representations.py``: Once the data has been split into sentence sets, BERT contextualized word representations of words in the stance datasets can be extracted with this script. Indicate the dataset with ``dataset_name``. They are saved in ``stance_split_data/``.

#### Calculating similarities
- ``calculate_comparison_similarity.py``: Once representations have been extracted, similarities can be calculated with this script. By default it is calculated on ``all`` datasets but you can change that with ``--dataset_name``. Indicate the vocabulary *V<sub>PQ</sub>* (Section 2.4) to be used with ``--vocab_for_sim`` (``all``, ``tfidf_10``, ``reversetfidf_10``). Indicate the BERT layer to be used with ``--layer``.


## Citation

If you use the code in this repository, please cite our paper:

```
@inproceedings{gari-soler-etal-2022-one,
    title = "One Word, Two Sides: Traces of Stance in Contextualized Word Representations",
    author = "Gar{\'\i} Soler, Aina  and
      Labeau, Matthieu  and
      Clavel, Chlo{\'e}",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.347",
    pages = "3950--3959",
}

```


## Contact

For any questions or requests feel free to [contact me](https://ainagari.github.io/menu/contact.html).


