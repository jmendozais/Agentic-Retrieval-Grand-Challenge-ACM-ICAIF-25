from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi
import faiss
from uuid import uuid4
from typing import List

import numpy as np


def create_hyde_query(query):
    return f"\
You are a finance expert. I want you to provide an answer well-structured, factual, detailed, and with examples if possible. \
Here is the only question you need to answer: {query}\n\
Answer:"


def reciprocal_rank_fusion(doc_idx_scores_svs, doc_idx_scores_kvs, top_k=4, semantic_weight = 0.5):

    def update_doc_ranks(doc_ranks, doc_idx_scores, k = 1, weight = 1):
        for rank, (doc_idx, _) in enumerate(doc_idx_scores):
            if doc_idx not in doc_ranks:
                doc_ranks[doc_idx] = 0
            doc_ranks[doc_idx] += weight / (rank + k)

    doc_ranks = {}
    if semantic_weight > 0:
        update_doc_ranks(doc_ranks, doc_idx_scores_svs, weight=semantic_weight)
    if semantic_weight < 1:
        update_doc_ranks(doc_ranks, doc_idx_scores_kvs, weight=1 - semantic_weight)

    ranked_docs = sorted(doc_ranks.items(), key=lambda x: x[1], reverse=True)
    results = [(doc_idx, score) for (doc_idx, score) in ranked_docs[:top_k]]
        
    return results


class BM25:

    def __init__(self, docs: List[str]):
        self.docs = docs
        tokenized_corpus = [doc.lower().split(" ") for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def similarity_search_with_relevance_scores(self, query, k=4):
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = scores.argsort()[::-1][:k]

        return [(i, scores[i]) for i in top_k_idx]
    

def create_faiss_vector_store(embedding_model, documents, distance_strategy): 
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
    normalize_L2 = distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE
    semantic_vs = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=normalize_L2,
        distance_strategy=distance_strategy
    )
    
    doc_objs = [Document(page_content=doc, metadata={"id": idx}) for idx, doc in enumerate(documents)]
    uuids = [str(uuid4()) for _ in range(len(doc_objs))]
    semantic_vs.add_documents(documents=doc_objs, ids=uuids)

    return semantic_vs


class HybridRetriever:

    def __init__(self, 
                 semantic_vs,
                 keyword_vs,
                 documents,
                 semantic_weight,
                 llm,
                 use_hyde=False,
                 ):
        self.semantic_vs = semantic_vs
        self.keyword_vs = keyword_vs
        self.semantic_weight = semantic_weight
        self.llm = llm
        self.use_hyde = use_hyde
        self.docs = documents

    def retrieve(self, query, top_k=20, top_k_candidates=20):
        try:
            doc_idx_scores_svs = None
            if self.semantic_weight > 0:
                doc_scores_svs = self.semantic_vs.similarity_search_with_relevance_scores(query, k=top_k_candidates)
                doc_idx_scores_svs = [(doc.metadata["id"], score) for doc, score in doc_scores_svs]
            
        except Exception as e:
            print("Exception during semantic search:")
            print(e)
        
        if self.semantic_weight < 1: 
            doc_idx_scores_kvs = self.keyword_vs.similarity_search_with_relevance_scores(query, k=top_k_candidates)

        if self.semantic_weight == 0.0:
            return doc_idx_scores_kvs[:top_k]
        elif self.semantic_weight == 1.0:
            return doc_idx_scores_svs[:top_k]
        else:
            return reciprocal_rank_fusion(doc_idx_scores_svs, 
                                          doc_idx_scores_kvs, 
                                          top_k=top_k,
                                          semantic_weight=self.semantic_weight)

    def rerank(self, query, docs_and_relevances, top_k):
        assert len(docs_and_relevances) >= top_k, "Number of documents to rerank must be >= top_k, \
            got {} < {}".format(len(docs_and_relevances), top_k)
        
        if self.use_hyde:
            hyde_prompt = create_hyde_query(query)
            hypothetical_answer = self.llm.invoke(hyde_prompt)
            hypothetical_answer = hypothetical_answer[len(hyde_prompt):].strip()
            query += "\n" + hypothetical_answer
        
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[hypothetical_answer, docs_and_relevances[i][0].page_content] for i in range(len(docs_and_relevances))]
        ce_scores = cross_encoder_model.predict(pairs)
        
        import numpy as np
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        ce_scores = sigmoid(ce_scores)

        combined_scores = []
        for i in range(len(docs_and_relevances)):
            doc, rank_score = docs_and_relevances[i]
            combined_score = 0.5 * rank_score + 0.5 * ce_scores[i]
            combined_scores.append((combined_score, doc))

        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in combined_scores[:top_k]]