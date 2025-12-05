import json
import random
import argparse
import os
import sys
import pandas as pd

"""
Generate candidate author samples for LLM authorship attribution.
The output is a JSON of the following format:
{
    test_sample_id: {
        query_author: 'author id',
        query_text: 'text of the test sample',
        candidate_authors: [list of candidate author ids],
        candidate_texts: [list of texts written by each of the authors]
    }
    ...

}
"""

SEED = 42

def group_df_by_author(df):
    """
    Group dataframe by author.

    Returns a dictionary where the key is the id and 
    value is a subset of the dataframe for the author id
    """
    return {str(author): group for author, group in df.groupby("labels")}

def sample_one_text_per_author(train_by_author, authors):
    """
    Randomly pick one text per author.
    Returns a dictionary, where the key is an author and value is a sampled text from the training set
    """
    sampled = {}
    for i, author in enumerate(authors):
        group = train_by_author[author]

        sampled_text = group["text"].sample(n = 1).iloc[0]
        sampled[author] = sampled_text
    return sampled


def build_candidate_sets(train_df, test_df):
    """
    Generate candidate author samples.
    Each dictionary entry consists of 
    - query_author: text example author id
    - query_text: test example text
    - candidate_authors: list of possible authors
    - candidate_texts: example texts from the training set for each author
    """
    random.seed(SEED)

    # Group train data by author
    train_by_author = group_df_by_author(train_df)
    authors = list(train_by_author.keys())

    candidate_sets = {}

    for pos, (_, example) in enumerate(test_df.iterrows()):
        query_author = example["labels"]
        query_text = example["text"]

        sampled_candidates = sample_one_text_per_author(train_by_author, authors)

        candidate_sets[pos] = {
            "query_author": query_author,
            "query_text": query_text,
            "candidate_authors": authors,
            "candidate_texts": sampled_candidates,
        }

    return candidate_sets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True, help="Path to training set (CSV file)")
    parser.add_argument("--test", required=True, help="Path to test set (CSV file)")
    parser.add_argument("--output", required=False, default="../data/candidate_sets.json",  help="Output JSON file")

    args = parser.parse_args()

    # Validate arguments
    train_exists = os.path.exists(args.train)
    test_exists = os.path.exists(args.test)
    if not train_exists and not test_exists:
        print(f"Error: neither train file '{args.train}' nor test file '{args.test}' exist.")
        sys.exit(1)

    out_path = args.output
    if os.path.isdir(out_path):
        print(f"Error: output path '{out_path}' is a directory.")
        sys.exit(1)

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        print(f"Warning: output file '{out_path}' exists and will be overwritten.")

    # Main logic
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    print("Building candidate sets...")
    candidate_sets = build_candidate_sets(train_df, test_df)

    print(f"Saving results to {out_path}")
    with open(out_path, "w") as f:
        json.dump(candidate_sets, f, indent=2)




