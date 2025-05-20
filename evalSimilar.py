import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel  
from scipy.spatial.distance import cdist    # for small-scale search
# import faiss    # for large-scale search
import os
from llm_utils import read_triples_from_file
from typing import List, Tuple, Dict
from logging import getLogger


logger = getLogger(__name__)
    
    
def emb_entities_list_w_sapbert(all_candidates: List[str]) -> torch.Tensor:
    """
    Calculate the embedding of all candidate entities (a list of entities) using SapBERT
    output is moved to cpu
    """
    # Example usage:
    # all_candidates = ["Cardiac complication", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 
    
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    
    # encode all candidate entities
    batchsize = 128 # batch size during inference
    all_candidate_embs = []
    for i in tqdm(np.arange(0, len(all_candidates), batchsize)):
        toks = tokenizer.batch_encode_plus(all_candidates[i:i+batchsize], 
                                           padding="max_length", 
                                           max_length=25, 
                                           truncation=True,
                                           return_tensors="pt").to('cuda')
        toks_cuda = {}
        for k,v in toks.items():
            toks_cuda[k] = v.cuda()
        candidate_cls_rep = model(**toks_cuda)[0][:,0,:]    # use CLS representation as the embedding
        all_candidate_embs.append(candidate_cls_rep.cpu().detach().numpy())
    all_candidate_embs = np.concatenate(all_candidate_embs, axis=0)
    # print(all_candidate_embs.shape)    # (len(all_candidates), 768)
    # print(all_candidate_embs)
    return all_candidate_embs

def emb_entity_str_w_sapbert(query: str) -> torch.Tensor:
    """
    Calculate the embedding of a query entity using SapBERT
    output is moved to cpu
    """
    # Example usage:
    # query = "cardiopathy"
    
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    
    # query entity to encode
    query_toks = tokenizer.batch_encode_plus([query], 
                                             padding="max_length", 
                                             max_length=25, 
                                             truncation=True,
                                             return_tensors="pt").to('cuda')
    query_emb = model(**query_toks)[0][:,0,:].cpu().detach().numpy()    # use CLS representation as the embedding
    # print(query_emb.shape)    # torch.Size([1, 768])
    return query_emb

def most_similar_entity_str_w_sapbert(query: str, all_candidates: List[str]) -> str:
    """
    Utilize the embedding of query entity and all candidate entities calculated using SapBERT, 
    and find the most similar entity to query.
    """
    # Example usage:
    # query = "cardiopathy"
    # all_candidates = ["Cardiac complication", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 
    
    query_emb = emb_entity_str_w_sapbert(query)
    all_candidate_embs = emb_entities_list_w_sapbert(all_candidates)

    # find query's nearest neighbour in all candidate
    dist = cdist(query_emb, all_candidate_embs)
    nn_index = np.argmin(dist)
    most_similar_entity = all_candidates[nn_index]
    print ("Predicted similar entity:", most_similar_entity)
    return most_similar_entity

def most_similar_entities_list_w_sapbert(queries: List[str], all_candidates: List[str]) -> List[str]:
    """
    Utilize the embeddings of query entities and all candidate entities calculated using SapBERT, 
    and find the most similar entity to each query.
    output the list of most similar entities for each query.
    """
    # Example usage:
    # queries = ["cardiopathy", "inflammation"]
    # print(f"queries: {queries}")
    # all_candidates = ["Cardiac complication", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"]
    # print(f"all_candidates: {all_candidates}")

    # Compute embeddings for all candidates once
    all_candidate_embs = emb_entities_list_w_sapbert(all_candidates)
    
    # Initialize list to store most similar entities
    most_similar_entities = []

    for query in queries:
        # Compute the embedding for the current query
        query_emb = emb_entity_str_w_sapbert(query)
        
        # Find the nearest neighbour in all candidate
        dist = cdist(query_emb, all_candidate_embs)  
        nn_index = np.argmin(dist)
        most_similar_entity = all_candidates[nn_index]
        most_similar_entities.append(most_similar_entity)
        # print("Predicted similar entity for '{}': {}".format(query, most_similar_entity))

    return most_similar_entities

def k_similar_entities_dict_w_sapbert(queries: List[str], all_candidates: List[str], k: int = 10) -> Dict[str, List[str]]:
    """
    Utilize the embeddings of query entities and all candidate entities calculated using SapBERT,
    and find the k most similar entities to each query.
    Output the dictionary of queries with their list of most similar entities.
    """
    # Example usage:
    # queries = ["cardiopathy", "inflammation"]
    # print(f"queries: {queries}")
    # all_candidates = ["Cardiac complication", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"]
    # print(f"all_candidates: {all_candidates}")
    # k = 2
    # print(f"k: {k}")

    # Compute embeddings for all candidates once
    all_candidate_embs = emb_entities_list_w_sapbert(all_candidates)
    
    # Determine actual number of candidates to return to avoid exceeding the number of candidates
    num_to_return = min(k, len(all_candidates))

    # Initialize dictionary to store the k most similar entities for each query
    k_similar_entities_dict = {}

    for query in queries:
        # Compute the embedding for the current query
        query_emb = emb_entity_str_w_sapbert(query)
        
        # Find the k nearest neighbour in all candidate
        dist = cdist(query_emb, all_candidate_embs)  
        nn_indices = np.argsort(dist.ravel())[:num_to_return]
        k_similar_entities = [all_candidates[idx] for idx in nn_indices]
        k_similar_entities_dict[query] = k_similar_entities
        # print("Predicted k most similar entities for '{}': {}".format(query, k_similar_entities))

    return k_similar_entities_dict

def get_k_similar_entities_dict_w_sapbert(not_matched_triples, dataset: str, k: int = 10) -> Dict[str, List[str]]:
    """
    Calculate k most similar entities for not matched triples using SapBERT.
    This might need large memory. 
    """
    # Extract all candidate entities from present KG triples for similarity evaluation
    gt_triples = read_triples_from_file(dataset, file_name='triples.txt')
    gt_entities = list(set(triple.replace('<', '').replace(">", '').split(', ')[0] for triple in gt_triples) | set(triple.replace('<', '').replace(">", '').split(', ')[2] for triple in gt_triples))
    
    query_entities = list(set(entity for triple in not_matched_triples for entity in (triple[0], triple[2])))
    k_similar_entities_dict = k_similar_entities_dict_w_sapbert(queries=query_entities, all_candidates=gt_entities, k=k)   # key: generated entity, value: list of k most similar gt entities

    return k_similar_entities_dict

# not used because of low performance
def eval_similar_w_sapbert(not_matched_triples: List[Tuple[str, str, str]], dataset: str) -> List[Tuple[str, str, str]]:
    """
    Find the most similar entity for each head and tail entity in not_matched_triples using SapBERT.
    Then find the most similar triple in gt_triples for each triple in not_matched_triples by ensuring that both the most similar head and tail entities appear in the gt_triple, irrespective of their positions.
    If found, add the triple to similar_gen_triples and the corresponding gt_triple to similar_gt_triples.
    """

    # Example usage:
    # not_matched_triples = [("cardiopathy", "has_symptom", "pain"), ("inflammation", "related_to", "swelling")]
    # not_matched_triples = [('PTK2', 'affects', 'plasma membrane H+'), ('PTK2', 'affects', 'NTA1 N-terminal amidase'), 
    #                        ('NTA1 N-terminal amidase', 'affects', 'protein degradation'), ('Protein degradation', 'affects', 'tolerance to heat stress in S'), 
    #                        ('Understanding exercise motivations', 'improves', 'health outcomes'), ('Exercise motivations', 'isa', 'health outcomes'), 
    #                        ('Exercise', 'isa', 'daily or recreational activity'), ('51', 'underwent', 'surgery'), 
    #                        ('Surgery', 'causes', 'pain'), ('Pain', 'isa', 'symptom'), 
    #                        ('Surgery', 'occurs in', 'hospital'), ('Hospital', 'isa', 'health care related organization'), 
    #                        ('Patient preference', 'isa', 'reason'), ('Bleeding', 'isa', 'injury or poisoning'), 
    #                        ('Causes', 'isa', 'reason'), ('Substance', 'isa', 'biologically active substance'), 
    #                        ('Geographic area', 'isa', 'location of'), ('Regulation or law', 'isa', 'governmental or regulatory activity')]
    
    similar_gen_triples = []
    similar_gt_triples = []
    
    # Extract all candidate entities from present KG triples
    gt_triples = read_triples_from_file(dataset, file_name='triples.txt')
    print(f"gt_triples: {len(gt_triples)}")
    gt_entities = list(set(triple.replace('<', '').replace(">", '').split(', ')[0] for triple in gt_triples) | set(triple.replace('<', '').replace(">", '').split(', ')[2] for triple in gt_triples))
    print(f"gt_entities: {len(gt_entities)}")

    # Extract all head and tail entities from not_matched_triples
    queries = [entity for triple in not_matched_triples for entity in (triple[0], triple[2])]
    print(f"not_matched_triples: {len(not_matched_triples)}")
    print(f"queries: {queries}")

    # Find most similar entities for all queries at once
    similar_entities = most_similar_entities_list_w_sapbert(queries, gt_entities)

    # Dictionary to map original query entities to their most similar entities
    entity_to_similar = dict(zip(queries, similar_entities))

    # Match not_matched_triples with gt_triples
    for head, rel, tail in not_matched_triples:
        print(f"Searching for similar triple for {(head, rel, tail)}...")
        similar_head = entity_to_similar[head]
        similar_tail = entity_to_similar[tail]
        
        # Find matching gt_triple
        for gt_triple in gt_triples:
            if similar_head in gt_triple and similar_tail in gt_triple and (head, rel, tail) not in similar_gen_triples:
                similar_gen_triples.append((head, rel, tail))
                similar_gt_triples.append(gt_triple)
    
    # logger.info(f"[STEP 6] Similar generated triples: {similar_gen_triples}")
    # logger.info(f"[STEP 6] Matching gt triples correspondingly: {similar_gt_triples}")
    print(f"{len(similar_gen_triples)} Similar Generated Triples: {similar_gen_triples}")
    print(f"Matching {len(similar_gt_triples)} GT Triples correspondingly: {similar_gt_triples}")
    return similar_gen_triples, similar_gt_triples


if __name__ == "__main__":
    # Example usage:
    not_matched_triples = [
        'N4_categories, decreases, AST',
        'N4_categories, decreases, ALT',
        'N4_categories, increases, total_protein',
        'N4_categories, increases, serum_albumin',
        'N4_categories, increases, globulin',
        'Random_forest_method, analyzes, environmental_variables'
    ]
    similar_triples_w_sapbert, _ = eval_similar_w_sapbert(not_matched_triples, 'UMLS')
    # similar_triples_w_promptlink, _ = eval_similar_w_promptlink(not_matched_triples, 'UMLS')