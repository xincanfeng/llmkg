import os
import random

'''
return a list of keywords from a file.
'''

def get_random_keywords(input_file, dataset, data_dir='./data'):
    input_file_path = os.path.join(data_dir, dataset, input_file)

    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    if not lines:
        print("No keywords available.")
        return
    
    random_keywords = random.choice(lines).strip().replace('\t', ', ')
    keywords = random_keywords.split(', ')

    print(f'Getting random keywords from: {input_file_path}')
    return keywords

def get_indexed_keywords(input_file, dataset, i, data_dir='./data'):
    input_file_path = os.path.join(data_dir, dataset, input_file)

    with open(input_file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]  
    
    if not lines:
        print("No keywords available.")
        return []
    
    index = i % len(lines)  
    
    selected_keywords = lines[index].replace('\t', ', ')
    keywords = selected_keywords.split(', ')

    print(f'Getting keywords from line {index+1} in: {input_file_path}')
    return keywords


if __name__ == "__main__":
    # keywords = get_random_keywords(dataset='BATMAN-TCM', input_file='hr_rt_ht.txt')
    keywords = get_indexed_keywords(dataset='UMLS', input_file='hr_rt_ht.txt', i=0)
    print(f"Selected keywords:\n{keywords}")