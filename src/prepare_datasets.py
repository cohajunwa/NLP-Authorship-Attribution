"""
Script for preparing the blog authorship corpus for authorship attribution
"""
import argparse
import os
import pandas as pd
import sys

from sklearn.model_selection import train_test_split

def preprocess(df, num_authors = 5):
    """
    Preprocess dataframe by
    - Truncating the dataset to include only top N authors (specified by num_authors)
    - Eliminating irrelevant columns
    - Cleaning blog text
    - Adjust labels

    Final dataframe contains top N authors (based on number of blog posts) and two columns
    - label: author id
    - text: text written by author
    """
    # Truncating the dataset
    df_count_by_author = df.groupby('id')['text'].count().sort_values(ascending = False)
    top_n_authors = df_count_by_author.head(num_authors).index.tolist()
    df = df[df['id'].isin(top_n_authors)]

    # Drop irrelevant columns
    df = df.drop(columns=['gender', 'age', 'topic', 'sign', 'date'])

    # Clean text (remove 'urlLink)
    df['text'] = df['text'].str.replace('urlLink', '', regex=False)

    # Adjust column names
    label2id = {author_id: idx for idx, author_id in enumerate(top_n_authors)} 
    df['labels'] = df['id'].map(label2id)
    df = df.drop(columns = ['id'])

    return df


def split_df(df):
    """
    Split dataframe into training, validation, and testing sets (80/10/10)
    """
    train, val_test = train_test_split(df, test_size=0.2) # Initial split (80% training, 20% testing + validation)
    val, test = train_test_split(val_test, test_size = 0.5) 

    return train, val, test
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--blogtext", required=True, help="Path to saved blog authorship corpus")

    args = parser.parse_args()

    raw_data_path = args.blogtext

    if not os.path.exists(raw_data_path):
        print("Dataset does not exist at this location!")
        sys.exit(1)

    raw_df = pd.read_csv(raw_data_path)
    df = preprocess(raw_df)
    train_df, val_df, test_df = split_df(df)

    print(f"Size of training set: {len(train_df)}")
    print(f"Size of validation set: {len(val_df)}")
    print(f"Size of test set: {len(test_df)}")

    print("Saving training, validation, and testing sets")
    train_df.to_csv('../data/blog_train.csv', index = False)
    val_df.to_csv('../data/blog_val.csv', index = False)
    test_df.to_csv('../data/blog_test.csv', index = False)