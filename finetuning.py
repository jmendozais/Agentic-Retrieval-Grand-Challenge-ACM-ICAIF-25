from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    datasets
    )
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

import peft
import datasets
import json
import os 
import tqdm

import dataset
import preprocessing


if __name__ == "__main__":

    # Create finetuning dataset if it does not exist
    dataset_dir = "/home/julio/Dataset/ACM-ICAIF-25/acm-icaif-25-ai-agentic-retrieval-grand-challenge"
    ft_dataset_path = dataset_dir + "/finetuning_data_chunk_dev.json"
    if os.path.exists(ft_dataset_path) is False:
        dsets = dataset.load_dataset(dataset_dir, [dataset.CHUNK_DEV_FNAME])
        dset = dsets[dataset.CHUNK_DEV_FNAME]
        dataset.create_dataset_for_finetuning(dset, ft_dataset_path)

    # Load finetuning dataset
    ft_dataset = datasets.load_dataset('json', data_files=ft_dataset_path, split='train')
        
    # Model set up
    model_kwargs = {"device":"cuda"}
    #embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_name = "FacebookAI/roberta-large"
    model = SentenceTransformer(model_name)

    # Set up PEFT
    lora_config = peft.LoraConfig(
        r=8, # Rank Number
        lora_alpha=32, # Alpha (Scaling Factor)
        lora_dropout=0.05, # Dropout Prob for Lora
        target_modules=["query", "key", "value"], # Only apply on MultiHead Attention Layer
        bias='none',
        task_type=peft.TaskType.FEATURE_EXTRACTION
    )
    model.add_adapter(lora_config)

    # Setup training
    loss = CoSENTLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/roberta-large-ft-wclean",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=4,#16,
        per_device_eval_batch_size=4,#16,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if your GPU can't handle FP16
        bf16=False,  # Set to True if your GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        #eval_strategy="steps",
        #eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="test",  # Used in W&B if `wandb` is installed
        gradient_accumulation_steps=4, # To reach a batch size of 16 with a constrained GPU memory
    )

    """
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )
    """

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ft_dataset,
        #eval_dataset=eval_dataset,
        loss=loss,
        #evaluator=dev_evaluator,
    )

    trainer.train()

    model.save_pretrained("models/roberta-large-ft-wclean-final")

