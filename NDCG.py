import json
import math
from collections import defaultdict
from argparse import ArgumentParser

def load_results_file(file_path):
    return json.loads(open(file_path, encoding='utf-8', mode='r').read())

def extract_benchmark(file_path):
    with open(file_path, encoding='utf-8', mode='r') as f:

        benchmarks = defaultdict(list)
        relevance_lookup = defaultdict(dict)

        for entry in f.read().split("\n"):
            entries = entry.split(" ")

            if entry and len(entries):

                query_id = entries[0]
                doc_name = entries[2]
                relevance_score = int(entries[-1])

                benchmarks[query_id].append({"name": doc_name, "relevance": relevance_score})
                relevance_lookup[query_id][doc_name] = relevance_score

        for key in benchmarks:
            benchmarks[key].sort(key=lambda x:(x["relevance"], x["name"]), reverse=True)

    return benchmarks, relevance_lookup

def set_resultant_grades(results, relevance_lookup_table):
    for query in results:
        for result in results[query]:
            doc_name = result["name"]
            look_up = relevance_lookup_table[query]

            result["relevance"] = look_up[doc_name] if doc_name in look_up else 0

    return results

def DCG(rankings, p):

    DCG = rankings[0]["relevance"]
    for i in range(1, p):
        if i >= len(rankings):
            break

        DCG += rankings[i]["relevance"] / math.log2(i+1)

    return DCG

def NDCG(rankings, benchmarks, p):

    dcg = DCG(rankings, p)
    idcg = DCG(benchmarks, p)

    return dcg/idcg

def avg_NDCG(all_results, all_benchmarks, p):

    avg = 0
    for query in all_results:
        curr = NDCG(all_results[query], all_benchmarks[query], p)
        print(query, f"{curr:.3f}")

        avg += curr

    avg = avg / len(all_results)
    print(f"Average NDCG for p = {p}: {avg:.4f}")

    return avg



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--benchmark', dest='benchmark', help='benchmark results file',
                        metavar='benchmark', required=True)
    parser.add_argument('--result', dest='results', help='personal results file',
                        metavar='results', required=True)
    parser.add_argument('--p', dest='p_value', help='p for DCG',
                        metavar='p', required=True)

    options = parser.parse_args()
    p = int(options.p_value)

    results = load_results_file(options.results)

    benchmarks, relevance_lookup_table = extract_benchmark(options.benchmark)

    results = set_resultant_grades(results, relevance_lookup_table)

    ans = avg_NDCG(results, benchmarks, p)

