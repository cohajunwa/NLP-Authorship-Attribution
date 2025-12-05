import argparse
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_model_output(model_output):
    """
    Parse model output. 
    If the output contains just a numberical value within the expected the range (as expected), return the value
    Otherwise, try to extract a numerical value. If the model response does not include a lable, return -1
    """

    try:
        pred = int(model_output)
    except ValueError:
        pred = -1

    return pred


def compute_metrics(df):
    preds = df['raw_model_output'].apply(parse_model_output)
    labels = df['true_author']
    accuracy = accuracy_score(labels, preds)

    # Macro metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # Micro metrics
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='micro', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'test_macro_precision': macro_precision,
        'test_macro_recall': macro_recall,
        'test_macro_f1': macro_f1,
        'test_micro_precision': micro_precision,
        'test_micro_recall': micro_recall,
        'test_micro_f1': micro_f1
    }

    for key, value in metrics.items():
        print(f"'{key}': {value},")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default="../data/llm_results.csv", help="Path to CSV file containing LLM results")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    compute_metrics(df)