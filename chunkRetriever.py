import numpy as np
import bm25s
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel
import json
import os
from logging import getLogger
import torch


logger = getLogger(__name__)


'''
More text preprocessing that can be considered:
lowercasing, stopword removal, stemming
'''

class ChunkRetriever:
    def __init__(self, retrieval_method, article_acquisition_mode, articles_file, read_articles_file, chunks_file, current_time, keywords_file_without_extension, log_file, dataset, save_dir):
        self.retrieval_method = retrieval_method
        self.log_dir = os.path.join(save_dir, dataset, f'{current_time}_{keywords_file_without_extension}_{log_file}')
        self.chunks_path = os.path.join(self.log_dir, chunks_file)
        if article_acquisition_mode == 'fetch':
            self.articles_dir = os.path.join(self.log_dir, articles_file)
        elif article_acquisition_mode == 'read':
            self.articles_dir = os.path.join('data', 'sources', read_articles_file)
        self.ensure_directories_exist()

        # Initialize the embedding model if using bge
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        # Setup the retrieval methods dictionary
        self.methods = {
            "bm25s": {
                "chunk_method": self.chunk_article_by_dot,
                "retrieval_model": self.bm25sRetriever
            },
            "rankbm25": {
                "chunk_method": self.chunk_article_by_dot,
                "retrieval_model": self.rankbm25Retriever
            },
            "bge": {
                "chunk_method": self.chunk_article_by_dot,
                "retrieval_model": self.bgeRetriever
            }
        }

    def ensure_directories_exist(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def _load_json_file(self):
        with open(self.articles_dir, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def chunk_article_by_dot(self, data):
        corpus = []
        for article in data:
            abstract = article.get("Abstract") or article.get("body") or ""
            sentences = [sentence.strip() for sentence in abstract.split('.') if sentence]
            corpus.extend(sentences)
        # print(corpus)
        return corpus

    def chunk_article_by_len(self, data, chunk_size=256, overlap=50):
        """从 JSON 数据中提取所有摘要,并将其按照tokens分割"""
        corpus = []
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("Chunk size must be larger than overlap size.")
        
        for article in data:
            abstract = article.get("Abstract", "")
            for i in range(0, len(abstract) - chunk_size + 1, step):
                chunk = abstract[i:i + chunk_size]
                corpus.append(chunk)
            
            if len(abstract) % step != 0:
                last_chunk = abstract[-chunk_size:]
                if last_chunk not in corpus:  
                    corpus.append(last_chunk)
        return corpus

    def query_with_all_keywords(self, keywords_list):
        """Combine keywords into a single string."""
        combined_keywords = ", ".join(keywords_list)  
        return combined_keywords

    def bm25sRetriever(self, corpus, query, top_k):
        """
        Retrieve top-k chunks using bm25s.
        Output: top-k chunks list in the format of [chunk str, chunk str, ...]
        Paper: https://arxiv.org/abs/2407.03618
        Code: https://huggingface.co/blog/xhluca/bm25s
        """
        top_k = min(top_k, len(corpus))  

        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(bm25s.tokenize(corpus))

        # Query the corpus and get top-k chunks
        top_k_chunks, scores = retriever.retrieve(bm25s.tokenize(query), k=top_k)
        top_k_chunks, scores = top_k_chunks[0], scores[0]
    
        print(f"Top-k chunks in the function:\n{top_k_chunks}")
        print(f"\nscores in the function:\n{scores}")
        return top_k_chunks

    def rankbm25Retriever(self, corpus, query, top_k):
        """
        Retrieve top-k chunks using BM25Okapi.
        Output: top-k chunks list in the format of [chunk str, chunk str, ...]
        Code: https://pypi.org/project/rank-bm25/
        """
        print(f"corpus length: {len(corpus)}")
        top_k = min(top_k, len(corpus))  

        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split(" ")
        top_k_chunks = bm25.get_top_n(tokenized_query, corpus, n=top_k)

        print(f"Top-k chunks in the function:\n{top_k_chunks}")
        return top_k_chunks
    
    def bgeRetriever_cpu(self, corpus, query, top_k):
        '''
        Retrieve top-k chunks using bge.
        
        :param corpus: List of strings.
        :param query: Single query string.
        :param top_k: Number of top chunks to retrieve.
        :return: List of top-k sentences with their similarity scores.
        Code: https://huggingface.co/BAAI/bge-m3
        '''
        top_k = min(top_k, len(corpus))  

        # Encode the query and the corpus
        query_embedding = self.model.encode([query],
                                       max_length=300, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                      )['dense_vecs']
        corpus_embeddings = self.model.encode(corpus)['dense_vecs']

        # Compute cosine similarities
        similarity = query_embedding @ corpus_embeddings.T

        # Get top-k indices based on similarity scores
        top_k_indices = similarity.argsort()[0][-top_k:][::-1]
        
        # Retrieve top-k sentences and their scores
        top_k_chunks = [corpus[i] for i in top_k_indices]

        return top_k_chunks

    def bgeRetriever(self, corpus, query, top_k, batch_size=1000000):
        '''
        Retrieve top-k chunks using bge in a batch-wise manner to handle large corpora.
        
        :param corpus: List of strings.
        :param query: Single query string.
        :param top_k: Number of top chunks to retrieve.
        :return: List of top-k sentences with their similarity scores.
        Code: https://huggingface.co/BAAI/bge-m3
        '''
        # Encode the query and the corpus
        query_embedding = self.model.encode([query],
                                       max_length=300, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                      )['dense_vecs']
        
        # Prepare to collect top-k results across batches
        all_scores = []
        all_indices = []

        # Process the corpus in batches
        for start_idx in range(0, len(corpus), batch_size):
            end_idx = min(start_idx + batch_size, len(corpus))
            batch_corpus = corpus[start_idx:end_idx]

            # Encode the batch of the corpus
            batch_embeddings = self.model.encode(batch_corpus)['dense_vecs']

            # Convert NumPy arrays to Torch Tensors and move to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            query_embedding_tensor = torch.tensor(query_embedding, device=device)
            batch_embeddings_tensor = torch.tensor(batch_embeddings, device=device)

            # Compute cosine similarities for the batch
            similarity = query_embedding_tensor @ batch_embeddings_tensor.T
            # Collect scores and corresponding indices
            top_scores, top_idx = torch.topk(similarity, min(top_k, len(batch_corpus)), largest=True, sorted=True)

            all_scores.extend(top_scores.cpu().numpy().flatten())
            all_indices.extend([idx + start_idx for idx in top_idx.cpu().numpy().flatten()])

        # Now, select the global top-k from all batches
        if len(all_scores) > top_k:
            # We only sort if we have more candidates than needed for top-k
            global_top_k_indices = np.argsort(all_scores)[-top_k:][::-1]
            final_indices = [all_indices[idx] for idx in global_top_k_indices]
            final_scores = [all_scores[idx] for idx in global_top_k_indices]
        else:
            final_indices = all_indices
            final_scores = all_scores

        # Retrieve final top-k sentences
        top_k_chunks = [corpus[i] for i in final_indices]

        return top_k_chunks
    
    def bgeRetriever_nobatch(self, corpus, query, top_k):
        '''
        Retrieve top-k chunks using bge.
        
        :param corpus: List of strings.
        :param query: Single query string.
        :param top_k: Number of top chunks to retrieve.
        :return: List of top-k sentences with their similarity scores.
        Code: https://huggingface.co/BAAI/bge-m3
        '''
        # from pudb import set_trace; set_trace()
        # print("BGE Model is on device:", self.model.device)

        # Encode the query and the corpus
        query_embedding = self.model.encode([query],
                                       max_length=300, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                      )['dense_vecs']
        corpus_embeddings = self.model.encode(corpus)['dense_vecs']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_embedding = torch.tensor(query_embedding, device=device)
        corpus_embeddings = torch.tensor(corpus_embeddings, device=device)

        # Compute cosine similarities
        similarity = query_embedding @ corpus_embeddings.T
        print(f"Similarity shape: {similarity.shape}")

        # Get top-k indices based on similarity scores
        sorted_indices = similarity.argsort(dim=1, descending=True)[0]  # Flatten and sort indices
        num_results = min(top_k, sorted_indices.size(0))  # Handle case where there are fewer results than top_k

        if num_results == 0:
            return []  # Return an empty list if there are no results

        top_k_indices = sorted_indices[:num_results]  # Fetch top-k indices safely

        # Retrieve top-k sentences and their scores
        top_k_chunks = [corpus[i] for i in top_k_indices]
        # top_k_scores = [float(similarity[0, i]) for i in top_k_indices]

        return top_k_chunks
    
    def _save_results(self, chunks):
        """Save the retrieved chunks to a file."""
        converted_chunks = []
        for chunk in chunks:
            if isinstance(chunk, np.ndarray):
                converted_chunks.append(chunk.tolist())
            else:
                converted_chunks.append(chunk)

        try:
            with open(self.chunks_path, 'w', encoding='utf-8') as f:
                json.dump(converted_chunks, f, ensure_ascii=False, indent=4)
            print(f"Retrieved chunks have been saved to {self.chunks_path}.")
        except Exception as e:
            print(f"Error saving results to {self.chunks_path}: {e}")
            raise

    def __call__(self, keywords, top_k=2):
        query = self.query_with_all_keywords(keywords)

        method_config = self.methods.get(self.retrieval_method, None)
        if not method_config:
            logger.error(f"[STEP 3] Invalid retrieval method: {self.retrieval_method}.")
            raise ValueError(f"[STEP 3] Invalid retrieval method: {self.retrieval_method}")

        corpus = method_config["chunk_method"](self._load_json_file())
        chunks = method_config["retrieval_model"](corpus=corpus, query=query, top_k=top_k)
        self._save_results(chunks)

        # print(f"query in the call: {query}")
        # print(f"chunks in the call: {chunks}")
        return chunks


if __name__ == "__main__":
    retriever = ChunkRetriever(
        retrieval_method='bm25s', 
        article_acquisition_mode='fetch', 
        articles_file='articles.json', 
        read_articles_file='pubmed.json', 
        chunks_file='chunks.json', 
        current_time='20241006', 
        keywords_file_without_extension='h_r_t', 
        log_file='log', 
        dataset='UMLS', 
        save_dir='./output'
        )
    keywords = ["herb", "protein"]
    result_chunks = retriever(keywords=keywords, top_k=2)

    