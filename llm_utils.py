import re
import os
from typing import List, Tuple, Dict
from logging import getLogger


logger = getLogger(__name__)


def read_triples_from_file(dataset: str, file_name: str = 'triples.txt', data_dir='./data') -> List[Tuple[str, str, str]]:
    """
    Read triples from a file and return a list of triples.
    Format requirement: one triple per line.
    All quotes and line breaks will be stripped. 
    """
    
    file_path = os.path.join(data_dir, dataset, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Use a list comprehension to read lines, remove all quotes, and skip empty lines
            triples = [line.replace("'", "").replace('"', "").strip() for line in file if line.strip()]
            print(f"Read {len(triples)} Triples from file {file_path}")
            return triples
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def parse_triples_from_file(dataset: str, file_name: str = 'triples.txt', data_dir='./data') -> List[Tuple[str, str, str]]:
    """
    Parse triples from a file and return a list of triples.
    Format requirement: Triples must exactly be in the format of <'head entity', 'relation', 'tail entity'>, ['head entity', 'tail entity', 'relation'], or ('head entity', 'tail entity', 'relation')
    Output: [('head', 'relation', 'tail'), ...] as what they are and as required when standardizing the triples.
    """

    # Regular expression pattern to capture triples
    pattern = re.compile(r'''
        [\[\(<]                    # Starting bracket (any of '[', '(', '<')
        \s*                        # Optional whitespace
        '(?P<head>[^']+?)'         # 'head' capturing everything inside single quotes
        \s*,\s*                    # Comma with optional whitespace
        '(?P<relation>[^']+?)'     # 'relation' capturing everything inside single quotes
        \s*,\s*                    # Comma with optional whitespace
        '(?P<tail>[^']+?)'         # 'tail' capturing everything inside single quotes
        \s*                        # Optional whitespace
        [\]\)>]                    # Ending bracket (any of ']', ')', '>')
    ''', re.VERBOSE | re.DOTALL)   # 're.DOTALL' to make '.' match newline characters as well

    triples = []
    file_path = os.path.join(data_dir, dataset, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                matches = pattern.finditer(line)
                for match in matches:
                    head, relation, tail = match.groups()
                    triples.append((head.strip(), relation.strip(), tail.strip()))
        # print(f"\n--- Parsing present KG triples ---\nParsed {len(triples)} Triples from file {file_path}:\n{triples}")
        logger.info(f"Parsed {len(triples)} present KG triples from file {file_path}")
        return triples
    except FileNotFoundError:
        logger.error(f"File {file_path} was not found.")
        return []
    except Exception as e:
        logger.error(f"An error occurred when parsing triples from file {file_path}: {e}")
        return []
    
def parse_triples_from_llm(kg_string: str, max_length: int = 80) -> List[Tuple[str, str, str]]:
    """
    Parse triples from a string generated by LLM and return a list of triples.
    Output: [(head, relation, tail), ...]
    """

    # Regular expression pattern to capture triples

    # match the pattern of two commas inside a pair of brackets: supporting angle brackets, parentheses and brackets, and supporting both English and Chinese commas (匹配成对的括号内含两个逗号的模式: 支持尖括号、圆括号和方括号，同时匹配中英文逗号)
    # pattern_easy = re.compile(r'[<\[\(]\s*(.*?)\s*[，,]\s*(.*?)\s*[，,]\s*(.*?)\s*[>\]\)]')

    pattern_quote = re.compile(r'''
        [\[\(<"']                  # Starting bracket or quote (any of '[', '(', '<', '"', "'")
        \s*                        # Optional whitespace
        '(?P<head>[^']+?)'         # 'head' capturing everything inside single quotes
        \s*[,，]\s*                 # Comma with optional whitespace, supporting both English and Chinese commas
        '(?P<relation>[^']+?)'     # 'relation' capturing everything inside single quotes
        \s*[,，]\s*                 # Comma with optional whitespace, supporting both English and Chinese commas
        '(?P<tail>[^']+?)'         # 'tail' capturing everything inside single quotes
        \s*                        # Optional whitespace
        [\]\)>"'|\n]               # Ending bracket or quote or newline (any of ']', ')', '>', '"', "'", '|', '\n')
    ''', re.VERBOSE | re.DOTALL)   # 're.DOTALL' to make '.' match newline characters as well

    pattern_comma = re.compile(r'''
        [\[\(<"']                  # Starting bracket or quote (any of '[', '(', '<', '"', "'")
        \s*                        # Optional whitespace
        (?P<head>[^,，]+?)          # 'head' capturing everything until the first comma (non-greedy), supporting both English and Chinese commas
        \s*[,，]\s*                 # Comma with optional whitespace, supporting both English and Chinese commas
        (?P<relation>[^,，]+?)      # 'relation' capturing up to the next comma (non-greedy), supporting both English and Chinese commas
        \s*[,，]\s*                 # Comma with optional whitespace, supporting both English and Chinese commas
        (?P<tail>.*?)              # 'tail' capturing everything until the matching closing bracket or quote (non-greedy)
        \s*                        # Optional whitespace
        [\]\)>|\n]                 # Ending bracket or newline (any of ']', ')', '>', '|', '\n')
    ''', re.VERBOSE | re.DOTALL)   # 're.DOTALL' to make '.' match newline characters as well

    kg_triples = []
    seen_triples = set()  # To avoid adding duplicates if both patterns match the same triple
    
    # Apply both patterns and add matches to the list
    for pattern in [pattern_quote, pattern_comma]:
        matches = pattern.finditer(kg_string)
        for match in matches:
            head = clean_symbols(match.group('head'))
            relation = clean_symbols(match.group('relation'))
            tail = clean_symbols(match.group('tail'))

            # Check if any part exceeds the maximum length
            if any(len(part) > max_length for part in (head, relation, tail)):
                continue  # Skip this triple

            triple = (head, relation, tail)  # formating the triple
            if triple not in seen_triples:
                seen_triples.add(triple)
                kg_triples.append(triple)

    print(f"\n--- Parsing generated kg triples ---\nParsed {len(kg_triples)} Triples from kg strings:\n{kg_triples}")
    return kg_triples

def parse_conflict_and_removal(text: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Parse the output text into two lists: conflicting triples and removal triples.
    """
    conflict_triples = []
    removal_triples = []

    # Normalize the text
    text = text.strip().lower()

    # Case 1: No conflicts
    if "no conflicting triples" in text or "conflicting triples: none" in text:
        conflict_triples = []
    else:
        conflict_text = text
        if "removal triples:" in conflict_text:
            conflict_text = conflict_text.split("removal triples:")[0]
        conflict_text = conflict_text.replace("conflicting triples:", "").strip()
        if "explanation:" in conflict_text:
            conflict_text = conflict_text.split("explanation:")[0].strip()
        conflict_triples = parse_triples_from_llm(conflict_text)

    # Case 2: No removals
    if "no triples to remove" in text or "removal triples: none" in text:
        removal_triples = []
    else:
        # Only look at text after "Removal Triples:"
        removal_text = text.split("removal triples:")[-1].strip()
        if "explanation:" in removal_text:
            removal_text = removal_text.split("explanation:")[0].strip()
        all_removal_triples = parse_triples_from_llm(removal_text)
        removal_triples = [t for t in all_removal_triples if t in conflict_triples]         # Keep only those removal triples also present in conflict_triples  

    return conflict_triples, removal_triples

def clean_symbols(text: str) -> str:
    """
    Remove specific unwanted symbols from the text and replace multiple spaces with a single space.
    """
    # Remove specific unwanted symbols
    text = text.replace('\n', '').replace("'", "").replace('"', '').replace('<', '').replace('>', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('_', ' ').replace(',', ' ')

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_entity(entity: str) -> str:
    """Clean entity by replacing underscores with spaces and standardizing whitespace."""
    # Replace underscores with spaces
    entity = entity.replace('_', ' ').replace("{", "").replace('}', '')

    # Remove 'the' and 'of' using regular expression, ignoring case
    # entity = re.sub(r"\b(the|of)\b", "", entity, flags=re.IGNORECASE)

    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    return re.sub(r"\s+", " ", entity).strip()

def clean_relation(relation: str) -> str:
    """
    Clean relation by replacing underscores with spaces, removing specific words ('of', 'the', 'is', 'for', 'has', 'to'),
    removing hyphens, and standardizing whitespace.
    """
    # Replace underscores with spaces to separate words
    relation = relation.replace('_', ' ').replace("{", "").replace('}', '')

    # Remove specific words and hyphens, applying case insensitive matching
    # relation = re.sub(r"\b(of|the|is|for|has|to)\b", "", relation, flags=re.IGNORECASE)

    # Replace multiple spaces with a single space and trim leading/trailing spaces
    return re.sub(r"\s+", " ", relation).strip()

def eval_match(kg_triples: List[Tuple[str, str, str]], dataset: str) -> List[Tuple[str, str, str]]:
    """
    Evaluate the match between the generated triples and the present KG triples using a simple string matching approach.
    """
    # remember to clean the gt triples file beforehand to save computation time
    cleaned_gt_triples = parse_triples_from_file(dataset)  # using triples.txt as the present KG triples file
    matched_triples = []
    not_matched_triples = []

    # for each kg_triple, clean its head entity, relation, and tail entity, and match it with the gt_triples
    for kg_triple in kg_triples:
        cleaned_kg_head = clean_entity(kg_triple[0])
        cleaned_kg_relation = clean_relation(kg_triple[1])
        cleaned_kg_tail = clean_entity(kg_triple[2])
        cleaned_kg_triple = (cleaned_kg_head, cleaned_kg_relation, cleaned_kg_tail)

        if cleaned_kg_triple in cleaned_gt_triples:
            matched_triples.append(cleaned_kg_triple)
        else:
            not_matched_triples.append(cleaned_kg_triple)

    print(f'\n--- Eval match --- \nMatched {len(matched_triples)} triples:\n{matched_triples}')
    print(f'\nNot matched {len(not_matched_triples)} triples:\n{not_matched_triples}')

    return matched_triples, not_matched_triples

def load_relation_types(dataset: str, file_name: str = "r.txt", data_dir: str = "./data") -> List[str]:
    """
    根据给定数据集名称，加载对应的关系类型列表。
    
    :param dataset: 数据集名称，用于定位目标目录。
    :return: 包含关系类型的列表。
    """
    relation_file_path = os.path.join(data_dir, dataset, file_name)
    if not os.path.exists(relation_file_path):
        raise FileNotFoundError(f"Relation file not found: {relation_file_path}")
    with open(relation_file_path, "r", encoding="utf-8") as f:
        relation_types = [line.strip() for line in f if line.strip()]
        print(f"relation_types: {relation_types}")
    return relation_types

def filter_triples_by_relations(triples: List[Tuple[str, str, str]], dataset: List[str]) -> List[Tuple[str, str, str]]:
    """
    Filters a list of triples based on a given list of relation types.
    """
    relation_types = load_relation_types(dataset)  # Load relation types from file
    filtered_triples = [triple for triple in triples if triple[1] in relation_types]
    return filtered_triples

def parse_keywords_from_llm(keyword_string: str) -> List[str]:
    """
    Parse a list of keywords from a string generated by LLM and return a clean list of keywords strings.
    """

    # Regular expression pattern to extract items inside a list-like structure
    pattern_quote = re.compile(r'''
        [\[\(<"']                  # Start with bracket or quote
        \s*                        # Optional whitespace
        '([^']+?)'                 # Single quoted entity
        \s*                        # Optional whitespace
        [\]\)>"'\n,]*              # End with bracket/quote/comma/newline (tolerate some misformat)
    ''', re.VERBOSE | re.DOTALL)

    pattern_comma = re.compile(r'''
        [\[\(<"']?                  # Optional starting bracket/quote
        \s*                         # Optional whitespace
        ([^,，\]\)>]+?)             # Entity text up to comma or bracket
        \s*                         # Optional whitespace
        [,，\]\)>]                  # Comma or closing bracket
    ''', re.VERBOSE | re.DOTALL)

    keywords = set()

    # Try matching quoted patterns first
    for pattern in [pattern_quote, pattern_comma]:
        matches = pattern.finditer(keyword_string)
        for match in matches:
            keyword = match.group(1)
            keyword = clean_entity(keyword)  # Using your clean_keyword function
            if keyword:
                keywords.add(keyword)

    keywords_list = list(keywords)
    print(f"\n--- Parsing generated keywords ---\nParsed {len(keywords_list)} keywords:\n{keywords_list}")
    return keywords_list