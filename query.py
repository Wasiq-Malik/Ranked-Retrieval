import json
from collections import defaultdict
from search import ranked_retrieval, okapi_tf, vector_space
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from argparse import ArgumentParser


def extract_queries(file_path):

    tree = ET.parse(file_path)

    root = tree.getroot()

    queries = {}

    for element in root:
        queries[element.attrib['number']] = element[0].text

    return queries

def generate_results(queries, score_function):

    result = defaultdict(list)
    for key in queries:
        scores = ranked_retrieval(queries[key], score_function)
        if scores is not None:
            # print top 3

            for i in range(len(scores)):
                #print("[Ranked-Retriever]", key, scores[i]["name"], i+1, scores[i]["score"])
                result[key].append({"name": scores[i]["name"], "rank": i+1, "score":scores[i]["score"]})

    return result



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--score', dest='score', help='name of scoring function (TF or TF-IDF)',
                        metavar='SCORE', required=True)

    parser.add_argument('--output', dest='output', help='name of the output file',
                        metavar='OUTPUT_FILE', required=True)

    options = parser.parse_args()
    score_function = options.score.lower()
    if score_function == "okapi-tf":
        score_function = okapi_tf
    elif score_function == "vector-space":
        score_function = vector_space
    else:
        print('Please select valid score function')
        exit(-1)

    queries = extract_queries("topics.xml")

    result = generate_results(queries, score_function)

    output_file = open(options.output, encoding='utf-8', mode='w')
    output_file.write(json.dumps(result))
