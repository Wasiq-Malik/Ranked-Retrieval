import os
import json
import time
import math
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from nltk.stem import snowball
from bs4.element import Comment
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize


def read_posting_list(file_pointer):

    posting_list = list()
    line = file_pointer.readline()
    tokens = [int(tok) for tok in line.split(",") if tok.isnumeric()]

    idx = 1
    gap = 0
    for _ in range(tokens[0]):
        posting = {"id": tokens[idx] + gap}   # read off doc id
        gap = posting["id"]
        idx += 1
        posting["freq"] = tokens[idx]   # read off word freq
        idx += 1
        posting["pos"] = [tokens[idx]]  # read off first position
        idx += 1

        for i in range(posting["freq"] - 1):
            # revert delta encoding
            posting["pos"].append(posting["pos"][i] + tokens[idx])
            idx += 1

        posting_list.append(posting)

    return posting_list


def load_docs_info(path):
    docs_info_file = open(path, encoding='utf-8', mode='r')
    return json.loads(docs_info_file.read())


def read_index_words(path):

    vocab = {}
    index_file = open(path, encoding='utf-8', mode='r')

    line = index_file.readline()
    while line != '':

        # insert word:byte_location pair in look-up table
        token = line.split(', ')
        vocab[token[0]] = int(token[1])
        line = index_file.readline()

    return vocab


def calc_L_avg():
    N = len(docs_info)
    avg = 0
    for i in range(N):
        avg += docs_info[str(i+1)]["length"]

    return avg / N


# globals
docs_info = load_docs_info('docs_meta_data.txt')
vocab = read_index_words('inverted_index_terms.txt')
L_avg = calc_L_avg()


def ranked_retrieval(query, score_function):

    print(f"\n[Ranked-Retriever] Querying '{query}' against index ... ")
    start_time = time.time()

    # tokenize query
    tokens = []
    for sentence in sent_tokenize(query):
        word_tokens = word_tokenize(sentence)
        tokens += word_tokens

    # clean query
    tokens = [tok.lower() for tok in tokens]

    # stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [tok for tok in tokens if tok not in stop_words]

    # stemming query tokens
    stemmer = snowball.SnowballStemmer('english')
    tokens = [stemmer.stem(tok) for tok in tokens]

    query_matches = defaultdict(list)
    with open("inverted_index_postings.txt", encoding='utf-8', mode='r') as posting_file:
        for word in tokens:
            if word in vocab:

                posting_loc = vocab[word]
                posting_file.seek(posting_loc)
                posting_list = read_posting_list(posting_file)

                df = len(posting_list)
                for posting in posting_list:
                    doc_id = str(posting["id"])

                    match = {"token": word}
                    match["df"] = df
                    match["tf"] = int(posting["freq"])

                    query_matches[doc_id].append(match)

    if len(query_matches):

        scores = score_function(tokens, query_matches)

        end_time = time.time()
        print(
            f"[Ranked-Retriever] Found {len(query_matches)} matches for '{query}' in {(end_time - start_time):.3f} seconds.")

        #for i in range(len(scores)):
        #    print("[Ranked-Retriever]", scores[i]["name"], i+1, scores[i]["score"])

        return scores

    else:
        print("[Boolean-Retriever] No Match Found.")

        return None


def okapi_tf(query, matches):

    k1 = 1.0
    b = 1.5
    N = len(docs_info)

    scores = []
    for id in matches:
        score = 0
        L_d = docs_info[id]["length"]
        name = docs_info[id]["path"].split("\\")[-1]
        for word in matches[id]:
            df = word["df"]
            tf = word["tf"]
            c_t = math.log((N - df + 0.5) / (df + 0.5))
            score += c_t * (((k1+1)*tf) / (k1*((1-b)+b*(L_d/L_avg))+tf))

        scores.append({"name": name.split(".")[0], "score": score})

    scores.sort(key=lambda x: x["score"], reverse=True)

    return scores


def vector_space(query, matches):
    query_tfs = {tok: 0 for tok in set(query)}

    # get weights for query vector
    for tok in query:
        query_tfs[tok] += 1

    # calculate magnitude for query vector
    q_mag = 0
    for tok in query_tfs:
        q_mag += query_tfs[tok] ** 2
    q_mag = q_mag ** (1/2)

    scores = []
    for id in matches:
        name = docs_info[id]["path"].split("\\")[-1]
        d_mag = docs_info[id]["magnitude"]
        dot_prod = 0

        for word in matches[id]:
            d_tf = word["tf"]
            q_tf = query_tfs[word["token"]]
            dot_prod += d_tf*q_tf

        score = dot_prod / (q_mag * d_mag)
        scores.append({"name": name.split(".")[0], "score": score})

    scores.sort(key=lambda x: (x["score"], x["name"]), reverse=True)

    return scores



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--score', dest='score', help='name of scoring function (TF or TF-IDF)',
                        metavar='SCORE', required=True)
    parser.add_argument('--query', dest='query', help='Search query',
                        metavar='QUERY', required=True)

    options = parser.parse_args()
    score_function = options.score.lower()
    if score_function == "okapi-tf":
        score_function = okapi_tf
    elif score_function == "vector-space":
        score_function = vector_space
    else:
        print('Please select valid score function')
        exit(-1)

    ranked_retrieval(options.query, score_function)
