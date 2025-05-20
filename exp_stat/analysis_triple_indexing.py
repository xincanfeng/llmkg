import csv

def load_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        return rows

def make_ent_rel_index(triple_file_path):
    # Load triples from CSV
    triples = load_csv(triple_file_path)

    # Initialize mappings and counters
    rel2id = {}
    id2rel = {}
    ent2id = {}
    id2ent = {}
    ent_num = 0
    rel_num = 0

    # Process each triple to populate dictionaries
    for head, relation, tail in triples:
        if relation not in rel2id:
            rel2id[relation] = rel_num
            id2rel[rel_num] = relation
            rel_num += 1
        if head not in ent2id:
            ent2id[head] = ent_num
            id2ent[ent_num] = head
            ent_num += 1
        if tail not in ent2id:
            ent2id[tail] = ent_num
            id2ent[ent_num] = tail
            ent_num += 1

    # Save entity and relation mappings to files
    with open('id2ent.txt', 'w') as f:
        for id, ent in id2ent.items():
            f.write(f"{id}\t{ent}\n")

    with open('id2rel.txt', 'w') as f:
        for id, rel in id2rel.items():
            f.write(f"{id}\t{rel}\n")

    # Optionally, print the number of entities and relations
    print('Number of entities:', ent_num)
    print('Number of relations:', rel_num)


if __name__ == '__main__':
    triple_file_path = 'test/UMLS/202410151811_h_r_t_F_rankbm25_pubmed/llama-3-8b_accumulated_not_match_but_correct.txt'
    make_ent_rel_index(triple_file_path)
