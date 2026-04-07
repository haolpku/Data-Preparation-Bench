
import os
import pickle
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI



class build_retriever:
    def __init__(self, model: str, top_k: int):
        self.project_path = os.environ["PROJECT_PATH"]
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        self.model = model
        self.top_k = top_k
        with open(f'{self.project_path}/evaluate/KB_Embedding/{self.model}_embeddings', 'rb') as fp:
            self.retriever = pickle.load(fp)
    
    def get_top_k_idx(self, scores):
        max_val = np.sort(scores)[-self.top_k]
        idx_list = np.where(scores>=max_val)[0].tolist()
        return idx_list
    
    def get_ada_embedding(self, query):
        embedding_model = "text-embedding-ada-002"
        query = query.replace("\n", " ")
        embed_vec = self.client.embeddings.create(input = [query], model=embedding_model).data[0].embedding
        return embed_vec

    def retrieve_idx_for_query(self, query):
        if self.model=='bm25':
            tokenized_query = query.split(" ")
            scores = self.retriever.get_scores(tokenized_query)
        elif self.model=='ada':
            embed_vec = self.get_ada_embedding(query)
            scores = np.dot(self.retriever, embed_vec)
        top_k_idx = self.get_top_k_idx(scores)
        return top_k_idx
