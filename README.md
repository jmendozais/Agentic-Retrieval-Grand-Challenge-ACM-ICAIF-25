# Agentic Retrieval Grand Challenge (ACM-ICAIF '25)

## Purpose
Run hybrid retrieval (BM25 + semantic embeddings) and produce reranked outputs for document- and chunk-level evaluation.

## Main files
- main.py — run reranking experiments (entry point).
- retriever.py — hybrid retriever implementation.
- finetuning.py — SentenceTransformer fine-tuning / PEFT (LoRA).
- dataset.py, preprocessing.py — dataset parsing and cleaning helpers.

## Requirements
- Python 3.9+ (ensure package compatibility)
- Install core packages:
  pip install -r requirements.txt

## Quick usage
1. Place dataset under a local path and update main.py or pass via CLI.
2. Ensure embedding model path is correct (main.py).
3. Run:
   python main.py
4. Outputs are written to the configured CSV/JSONL files.
