import json
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Union, Tuple

####################
# Data Manipulation 
####################

# Function to load data
def load_data(
    file_path: str, index_col: str = "stimulus_id", encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Arguments
    ---------
    file_path : str
        The path to the file to load
    index_col : str
        The column to use as the index
    encoding : str
        The encoding to use when reading the file

    Returns
    -------
    pd.DataFrame
        The loaded data
    """
    data = pd.read_csv(file_path, index_col=index_col, encoding=encoding)
    return data


###################
# Vector DB utility
###################


def convert_df_to_documents(
    df: pd.DataFrame, text_column: str, themes_definitions: dict
) -> Tuple[List[Document], List[int]]:
    """
    Arguments
    =========
    df: pd.DataFrame
        DataFrame to be converted
    text_column: str
        the column containing the actual text 
        (it's good practice to delete datapoints with empty text)
    themes_definitions: dict
        what is defined at definitions/themes.json

    Returns
    =========
    tuple
        docs, ids
    """
    docs = []
    ids = []

    docs = []
    ids = []

    for datapoint_id, row in df.iterrows():
        dictrow = row.to_dict()
        transcript = dictrow.pop(text_column)

        for theme in themes_definitions.keys():
            # replace NaN in the empty themes cues with the value "-1"
            if pd.isna(dictrow[theme + "_cues"]):
                dictrow[theme + "_cues"] = -1

        dictrow["id"] = datapoint_id
        docs.append(Document(page_content=transcript, metadata=dictrow))
        ids.append(datapoint_id)

    return docs, ids


######################
# Get Prompt Templates
######################


def get_prompt(template: str) -> Tuple[PromptTemplate, dict, Union[dict, None]]:
    """
    Arguments
    =========
    template: str
        The template to use for prompting the LLM. This is a string
        indicating the name of the corresponding subfolder in the prompts folder
        which in turn should contain a prompt.txt file indicating the prompt
        and a inputs.json file indicating the input variables to the prompt (see the provided examples)

    Returns
    =========
    PromptTemplate
        The langachin prompt template based on the provided template.
    """

    with open(os.path.join("prompts", template, "prompt.txt")) as f:
        prompt = "\n".join(f.readlines())

    with open(os.path.join("prompts", template, "inputs.json")) as f:
        inputs = json.load(f)

    with open(os.path.join("prompts", template, "placeholder.json")) as f:
        placeholder = json.load(f)

    # Check if the reasoning_structure input variable is needed and if the file exists
    if "{reasoning_structure}" in prompt:
        if "reasoning_structure" not in inputs["input_variables"]:
            inputs["input_variables"].append("reasoning_structure")
        if not os.path.exists(os.path.join("prompts", template, "reasoning_structure.json")):
            raise FileNotFoundError(
                f"Your prompt for {template} requires you to define {os.path.join('prompts', template, 'reasoning_structure.json')}"
            )
        with open(os.path.join("prompts", template, "reasoning_structure.json")) as f:
            reasoning_structure = json.load(f)

        return PromptTemplate(template=prompt, input_variables=inputs), placeholder, reasoning_structure

    return PromptTemplate(template=prompt, input_variables=inputs), placeholder


def get_top_classes_per_theme(df, classes_key, themes_definitions, break_ratio=1.2):

    if break_ratio <= 1:
        raise ValueError("break_ratio must be greater than 1")

    balanced_dfs = []
    counts = df[classes_key].value_counts().to_dict()
    for target in counts.keys():
        temp = df[df[classes_key] == target]
        balanced_dfs.append(temp.sample(counts[min(counts)], random_state=42))
    balanced_df = pd.concat(balanced_dfs)

    predominant_classes = {}

    for theme in themes_definitions.keys():
        counts = balanced_df[[classes_key,theme+"_cues"]].groupby(classes_key).count()
        counts = counts[theme+"_cues"].to_dict()  
        descending = sorted(counts, key=counts.get)[::-1]               

        # find break point in frequency (`break_ratio` times the next class)
        predominant_classes[theme] = []
        for i, cls_ in enumerate(descending):
            predominant_classes[theme].append(cls_)
            if i != len(descending) - 1:
                if counts[cls_] >= break_ratio * counts[descending[i+1]]:
                    break
        else:
            # this theme has no predominant classes
            # return empty list
            predominant_classes[theme] = []
                
    return predominant_classes

####################
# Baseline Functions
####################


######
# TODO: Implement setfit 
# 1. text -> setfit -> themes (individually)
# 2. text + themes -> setfit -> target_classification
# 3. text + themes + audio_description -> setfit -> target_classification
######
