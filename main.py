import tqdm
import json
import argparse

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import dataset
from retrieve import HybridRetriever, BM25, create_faiss_vector_store
from preprocessing import clean_text
import exp_config


DOCUMENT_TYPES = ['DEF14A', '10-K', '10-Q', '8-K', 'Earnings']
DOC_TYPE_DESC_FILE = "data/doc_type_descriptions.json"


def rerank_dataset(data, 
                   output_path="reranked_results.jsonl", 
                   use_doc_type_desc = True,
                   is_doc_rank_task = True,
                   semantic_weight = 0,
                   distance_strategy = None):
    
    model_kwargs = {"device":"cuda"}
    embedding_model_name = "/home/julio/Documents/professional-skills/ml/ICAIF25/models/roberta-large-ft/checkpoint-4200"

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, 
                                       model_kwargs=model_kwargs, 
                                       encode_kwargs={'normalize_embeddings':True})
    
    if is_doc_rank_task:
        if use_doc_type_desc:
            doc_type_descs_lst = []
            with open(DOC_TYPE_DESC_FILE, "r", encoding="UTF-8") as f:
                doc_type_descs = json.load(f)
                for doc_type in DOCUMENT_TYPES:
                    doc_type_descs_lst.append(doc_type_descs[doc_type])
        else:
            doc_type_descs_lst = DOCUMENT_TYPES

        # Clean data
        doc_type_descs_lst = [clean_text(doc) for doc in doc_type_descs_lst]

        semantic_vs = None
        keyword_vs = None
        if semantic_weight < 1:
            keyword_vs = BM25(doc_type_descs_lst)
        if semantic_weight > 0:
            semantic_vs = create_faiss_vector_store(embedding_model, 
                                                    doc_type_descs_lst, 
                                                    distance_strategy=distance_strategy) 

    with open(output_path, "a", encoding="utf-8") as out_file:
        for i, entry in tqdm.tqdm(enumerate(data)):
            if len(entry["messages"]) == 0:
                raise ValueError(f"No messages found in the entry {i}")

            documents = []
            id, query, documents = dataset.parse_dataset_line(entry, is_doc_rank_task=is_doc_rank_task).values()

            if len(documents) == 0:
                raise ValueError(f"No documents found in the entry {i}")
            
            # Clean data
            documents = [clean_text(doc) for doc in documents]

            if not is_doc_rank_task:        
                semantic_vs = None
                keyword_vs = None
                if semantic_weight < 1:
                    keyword_vs = BM25(documents)
                if semantic_weight > 0:
                    semantic_vs = create_faiss_vector_store(embedding_model, 
                                                            documents,
                                                            distance_strategy=distance_strategy) 
                    
            retriever = HybridRetriever(semantic_vs=semantic_vs,
                                        keyword_vs=keyword_vs,
                                        documents=documents,
                                        semantic_weight=semantic_weight,
                                        llm=None,
                                        use_hyde=False)
            
            doc_scores = retriever.retrieve(query, 
                                            top_k=5, 
                                            top_k_candidates=25)
            for doc_idx, _ in doc_scores:
                out_file.write(f"{id},{doc_idx}\n")
        

def run_config(datasets, 
               semantic_weight, 
               distance_strategy, 
               output_path):
    
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write("sample_id,target_index\n")

    rerank_dataset(datasets[dataset.DOC_EVAL_FNAME], 
                output_path=output_path, 
                is_doc_rank_task=True,
                semantic_weight=semantic_weight,
                distance_strategy=distance_strategy
                )
    
    rerank_dataset(datasets[dataset.CHUNK_EVAL_FNAME], 
                output_path=output_path, 
                is_doc_rank_task=False,
                semantic_weight=semantic_weight,
                distance_strategy=distance_strategy
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank documents using hybrid retrieval.")
    parser.add_argument("--dataset_dir", 
                        type=str, 
                        default="/home/julio/Dataset/ACM-ICAIF-25/acm-icaif-25-ai-agentic-retrieval-grand-challenge")
    parser.add_argument("--exp_config_key", 
                        type=str, 
                        default="semantic_weight")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    exp_configs = exp_config.get_exp_configs(args.exp_config_key)

    datasets = dataset.load_dataset(dataset_dir)

    for i in range(len(exp_configs)): # You can change this to a range if needed
        run_config(datasets, **exp_configs[i])