from collections import defaultdict
import os
import json


def read_triples_from_file(triples_file_path):
    try:
        with open(triples_file_path, 'r', encoding='utf-8') as file:
            # Use a list comprehension to read lines, remove all quotes, and skip empty lines
            triples = [line.replace("'", "").replace('"', "").strip() for line in file if line.strip()]
            print(f"Read {len(triples)} Triples from file {triples_file_path}")
            return triples
    except FileNotFoundError:
        print(f"Error: The file {triples_file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def triple_statistics(triples_file_path):
    triples = read_triples_from_file(triples_file_path)

    entities = set()
    relations = set()
    entity_relations = defaultdict(set)

    for triple in triples:
        e1, r, e2 = triple.replace('<', '').replace(">", '').split(', ')
        entities.update([e1, e2])
        relations.add(r)
        entity_relations[e1].add(r)
        entity_relations[e2].add(r)

    total_entities = len(entities)
    total_relations = len(relations)

    relation_count = defaultdict(int)
    for entity, rels in entity_relations.items():
        relation_count[len(rels)] += 1

    # Prepare output content
    output = []
    output.append(f"Total number of entities: {total_entities}")
    output.append(f"Total number of relations: {total_relations}")
    print(f"Total number of entities: {total_entities}")
    print(f"Total number of relations: {total_relations}")

    for i in range(1, 20):
        count = relation_count[i]
        percentage = (count / total_entities) * 100 if total_entities > 0 else 0
        output.append(f"Entities with exactly {i} relations: {count} ({percentage:.2f}%)")
        print(f"Entities with exactly {i} relations: {count} ({percentage:.2f}%)")

    with open('metrics_triples.txt', 'w') as f:
        for line in output:
            f.write(line + '\n')

    return

def read_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

def calculate_metrics(json_file_path, gold_standard):
    data = read_json(json_file_path)
    if not data:
        return  # Exit if the data couldn't be read or was empty
    
    results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0})
    
    # Iterate through each iteration of evaluations
    for entry in data:
        gold_answers = set(tuple(triple) for triple in entry['evaluations'][gold_standard]['not_match_but_correct'])
        
        for model, evaluation in entry['evaluations'].items():
            model_answers = set(tuple(triple) for triple in evaluation['not_match_but_correct'])
            
            tp = len(model_answers & gold_answers)
            fp = len(model_answers - gold_answers)
            fn = len(gold_answers - model_answers)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results[model]['tp'] += tp
            results[model]['fp'] += fp
            results[model]['fn'] += fn
            results[model]['precision'] = precision
            results[model]['recall'] = recall
    
    with open('metrics_recall.txt', 'w') as file:
        for model, metrics in results.items():
            file.write(f"Model: {model}, Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}\n")
            print(f"Model: {model}, Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}")

    return results


if __name__ == '__main__':

    # Example usage 1:
    triples_file_path = os.path.join('data', 'UMLS', 'triples.txt')
    triple_statistics(triples_file_path=triples_file_path)


    # Example usage 2:
    json_file_path = 'test/UMLS/202410151811_h_r_t_F_rankbm25_pubmed/combined_results.json'  # Modify with the correct file path
    calculate_metrics(json_file_path=json_file_path, gold_standard='mistral-7b')

    