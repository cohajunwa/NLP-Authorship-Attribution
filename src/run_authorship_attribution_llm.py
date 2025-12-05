import argparse
import json
import os
import pandas as pd
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from jinja2 import Environment, FileSystemLoader

load_dotenv()

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
SYSTEM_PROMPT = """Respond with ONLY the label. Do not include additional text. The label MUST be among the provided potential authors."""

CLIENT = genai.Client(api_key=GEMINI_API_KEY)

def generate_model_output(prompt: str, model: str):
    """
    Generate model output (using exponential backoff strategy to mitigate rate limit errors)

    Args:
        prompt (str): Prompt
        model (str): Gemini model name

    Returns:
        response (str)
    """
    num_retries = 0
    delay = 1
    max_retries = 10

    while True:
        try:
            response = CLIENT.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT),
                contents=prompt
            )

            return response.text
        
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"Retrying after encountering rate limit error: {e}")
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= 2 * (1 + random.random())
                time.sleep(delay)
            else:
                raise e

        except Exception as e:
            raise(e)



def construct_prompt(template, query_text, example_texts):
    return template.render(
        query_text = query_text,
        example_texts = example_texts
    )

def collect_results(prompt_template, candidate_sets, model: str):
    """Workflow for authorship attribution
    For each entry in the test set sample
    1. Construct a prompt
    2. Generate a model response

    Args:
        prompt_template: Jinja2 template for prompts
        candidate_sets: Dictionary of candidate sets
        model (str): Gemini model name

    Return a dataset containing the model's response and the correct response
    """
    data = []

    for idx, entry in candidate_sets.items():
        print(f'Processing test example {idx}')
        true_author = entry['query_author']

        query_text = entry['query_text']
        example_texts = entry['candidate_texts']

        prompt = construct_prompt(prompt_template, query_text, example_texts)

        try:
            raw_output = generate_model_output(prompt, model)
        except Exception as e:
            print(f"Error occurred while processing: {e}")
            raw_output = ''
        print(f"Model's response: {raw_output}")
        print(f"Correct label: {true_author}\n")

        data.append({
            'input_text': query_text,
            'true_author': true_author,
            'raw_model_output': raw_output,
        })

        # time.sleep(10) # To avoid reaching requests per min limit

    return pd.DataFrame(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_sets", required=False, default = "../data/candidate_sets.json", help="Path to JSON file containing candidate sets")
    parser.add_argument("--output", required=False, default="../data/llm_results.csv")
    parser.add_argument("--model", required=False, default="gemini-2.0-flash-lite", help="Gemini model to use (default: gemini-2.0-flash-lite)")

    args = parser.parse_args()

    jinja_env = Environment(loader=FileSystemLoader('../prompts'))
    template = jinja_env.get_template('lip_prompt.txt')

    if not os.path.exists(args.candidate_sets):
        print(f"Candidate set JSON file does not exist at this location!")

    candidate_sets = json.load(open(args.candidate_sets))
    df = collect_results(template, candidate_sets, args.model)
    print(f'Saving results to CSV')
    df.to_csv(args.output)