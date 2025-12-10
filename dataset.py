import os
import json
import re
import tqdm

import preprocessing

# Dataset file names 
DOC_DEV_FNAME = "document_ranking_kaggle_dev.jsonl"
DOC_EVAL_FNAME = "document_ranking_kaggle_eval.jsonl"
CHUNK_DEV_FNAME = "chunk_ranking_kaggle_dev.jsonl"
CHUNK_EVAL_FNAME = "chunk_ranking_kaggle_eval.jsonl"
ALL_DATA_FNAMES = [DOC_DEV_FNAME, DOC_EVAL_FNAME, CHUNK_DEV_FNAME, CHUNK_EVAL_FNAME]


def load_dataset(dataset_dir, data_fnames = ALL_DATA_FNAMES):
    datasets = {}

    for fname in data_fnames:
        path = os.path.join(dataset_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            datasets[fname] = [json.loads(line) for line in f]
            
    return datasets


def parse_dataset_line(entry, is_doc_rank_task=True):
    # Extract id
    if "_id" in entry: 
        id = entry["_id"]
        
    elif "uuid" in entry:
        id = entry["uuid"]

    documents = []
    content = entry["messages"][0]["content"]

    if is_doc_rank_task:
        # Extract query
        match = re.search(r"Question:\s*(.*?)\s*Document Types to rank:", content, re.DOTALL | re.IGNORECASE)
        assert match is not None, "Query not found in the content."
        query = match.group(1).strip()

        # Define pattern to extract documents
        pattern = re.compile(r"\[Document Index \d+\]\s*(.*?)(?=\[Document Index \d+\]|\Z)", re.DOTALL)
    else:
        # Extract query
        match = re.search(r"Question:\s*(.*?)\s*Text chunks:", content, re.DOTALL | re.IGNORECASE)
        assert match is not None, "Query not found in the content."
        query = match.group(1).strip()

        # Define pattern to extract chunks
        pattern = re.compile(r"\[Chunk Index \d+\]\s*(.*?)(?=\[Chunk Index \d+\]|\Z)", re.DOTALL)
    
    documents = []
    for match in pattern.finditer(content):
        documents.append(match.group(1).strip())

    assert len(documents) > 0, "No documents found in the content."

    parsed_entry = {
        "id": id,
        "query": query,
        "documents": documents
    }

    # Extract relevance scores if available
    if "qrel" in entry.keys():
        assert len(entry["qrel"]) == len(documents), f"len qrel {len(entry['qrel'])} != len documents {len(documents)}"
        parsed_entry["scores"] = [entry["qrel"][str(i)] for i in range(len(documents))]

    return parsed_entry


def create_dataset_for_finetuning(dset, output_path):   

    with open(output_path, 'w') as f_out:
        for entry in tqdm.tqdm(dset):
            parsed_entry = parse_dataset_line(entry, is_doc_rank_task=False)
            # Do something with id, query, documents

            for i, doc in enumerate(parsed_entry["documents"]):
                if len(doc) == 0:
                    print("Dropping an empty document ", parsed_entry["query"], i)
                else: 
                    ft_entry = {}
                    ft_entry["sentence1"] = preprocessing.clean_text(parsed_entry["query"])
                    ft_entry["sentence2"] = preprocessing.clean_text(doc)
                    ft_entry["label"] = float(parsed_entry["scores"][i])
                    f_out.write(json.dumps(ft_entry) + "\n")