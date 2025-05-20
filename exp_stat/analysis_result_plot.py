import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the tokenizer and model for SapBERT
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  

def read_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)
    
def get_embedding(text):
    """Compute the embedding of a given text using SapBERT."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}          # Move input tensors to the same device as the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

def compute_distances(key_entity, entities_list):
    """Compute cosine distances between the key entity and a list of entities."""
    key_embedding = get_embedding(key_entity)
    distances = []
    for entity in entities_list:
        entity_embedding = get_embedding(entity)
        similarity = cosine_similarity(
            key_embedding.reshape(1, -1),
            entity_embedding.reshape(1, -1)
        )[0][0]
        distance = 1 - similarity  # Convert similarity to distance
        distances.append(distance)
    return distances

def analyze_models(data):
    """Analyze different models' filtering capabilities."""
    iteration_data = data[0]  # Assuming one iteration as per the sample
    keywords = iteration_data['keywords']
    initial_keyword = keywords[0]  # Assuming one keyword

    k_similar_entities_dict = iteration_data['generated_data']['k_similar_entities_dict']
    evaluations = iteration_data['evaluations']

    results = {}

    for model_name, model_eval in evaluations.items():
        print(f"Analyzing model: {model_name}")
        model_results = {}
        similar_entities_dict = model_eval.get('similar_entities_dict', {})
        total_keys = len(k_similar_entities_dict)
        total_filtered_entities = 0
        total_entities = 0

        for key_entity in k_similar_entities_dict.keys():
            original_entities = k_similar_entities_dict[key_entity]
            filtered_entities = similar_entities_dict.get(key_entity, [])
            num_original = len(original_entities)
            num_filtered = len(filtered_entities)
            num_filtered_out = num_original - num_filtered

            total_entities += num_original
            total_filtered_entities += num_filtered_out

            # Compute distances for original and filtered entities
            distances_original = compute_distances(key_entity, original_entities)
            distances_filtered = compute_distances(key_entity, filtered_entities) if filtered_entities else []
            # Compute distances to the initial keyword
            distances_to_keyword = compute_distances(initial_keyword, original_entities)

            # Store the computed distances
            model_results[key_entity] = {
                'num_original': num_original,
                'num_filtered': num_filtered,
                'num_filtered_out': num_filtered_out,
                'distances_original': distances_original,
                'distances_filtered': distances_filtered,
                'distances_to_keyword': distances_to_keyword,
                'original_entities': original_entities,
                'filtered_entities': filtered_entities
            }

        # Calculate overall filtering statistics
        percentage_filtered = (total_filtered_entities / total_entities) * 100 if total_entities > 0 else 0
        results[model_name] = {
            'model_results': model_results,
            'total_entities': total_entities,
            'total_filtered_entities': total_filtered_entities,
            'percentage_filtered': percentage_filtered
        }

    return results

def plot_entity_distances(model_results, model_name):
    """Plot distances between entities for visualization."""
    for key_entity, result in model_results.items():
        entities = result['original_entities']
        distances = result['distances_original']
        plt.figure(figsize=(10, 6))
        plt.barh(entities, distances)
        plt.xlabel('Cosine Distance')
        plt.title(f"Distances between '{key_entity}' and its similar entities ({model_name})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

def get_triple_embedding(triple):
    """Compute the embedding of a triple by concatenating its elements."""
    triple_text = ' '.join(triple)
    return get_embedding(triple_text)

def compute_triple_distances(triples_list):
    """Compute pairwise cosine distances between triples."""
    embeddings = [get_triple_embedding(triple) for triple in triples_list]
    similarities = cosine_similarity(embeddings)
    distances = 1 - similarities  # Convert similarities to distances
    return distances

if __name__ == '__main__':
    json_file_path = 'test/UMLS/202410151811_h_r_t_F_rankbm25_pubmed/combined_results.json'  
    data = read_json(json_file_path)
    results = analyze_models(data)

    # Example of plotting for one model
    for model_name, model_data in results.items():
        print(f"Model: {model_name}")
        print(f"Total entities: {model_data['total_entities']}")
        print(f"Total filtered out: {model_data['total_filtered_entities']}")
        print(f"Percentage filtered: {model_data['percentage_filtered']:.2f}%\n")
        plot_entity_distances(model_data['model_results'], model_name)