from jinja2 import Environment, BaseLoader
# from jinja2 import Template
import json
import os
from logging import getLogger


logger = getLogger(__name__)


def read_file_content(file_name, dataset, data_dir='./data'):
    """Reads file and returns its content, returns None if file does not exist."""
    file_path = os.path.join(data_dir, dataset, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError as e:
        logger.error(f"{file_path} not found: {e}")
        return None
    
def read_file_and_process_lines(file_name, dataset, data_dir='./data'):
    """Read lines from a file and return them as a comma-separated string."""
    file_path = os.path.join(data_dir, dataset, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
        list = ', '.join(line.strip() for line in lines)
        return list
    except FileNotFoundError as e:
        logger.error(f"{file_path} not found: {e}")
        return None
    
def read_json_data(file_path):
    """Reads JSON data from a specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading JSON from {file_path}: {e}")
        return None
    
def jinjaReadCriteria(keywords_expander, dataset):
    # Read data for rendering
    if keywords_expander == 'none' or keywords_expander == 'base':
        purposes = read_file_content('purposes.txt', dataset)
        entity_types = read_file_and_process_lines('entity_types.txt', dataset)
    elif keywords_expander == 'llama' or keywords_expander == 'mistral':
        purposes = read_file_content(f'purposes_{keywords_expander}.txt', dataset)
        entity_types = read_file_and_process_lines(f'entity_types_{keywords_expander}.txt', dataset)     
    else:
        logger.error(f"Invalid keywords_expander: {keywords_expander}")

    an_example = read_file_content('an_example.txt', dataset)
    more_examples = read_file_and_process_lines('more_examples.txt', dataset)
    entity_list = read_file_and_process_lines('h_t.txt', dataset)
    relation_list = read_file_and_process_lines('r.txt', dataset)
    criteria_json = read_json_data('criteria.json')
    # print("purposes:", purposes)
    # print("Entity List:", entity_list)
    # print("Relation List:", relation_list)

    if not all([purposes, an_example, more_examples, entity_types, entity_list, relation_list, criteria_json]):
        logger.error("One or more required files for creating specific criteria do not exist.")
        return

    # Create a single Jinja Environment
    env = Environment(loader=BaseLoader())

    try:
        # Generate_kg template rendering
        generate_kg_criteria_string = json.dumps(criteria_json['Generate_kg'], indent=4, ensure_ascii=False) 
        generate_kg_criteria_template = env.from_string(generate_kg_criteria_string)
        rendered_generate_kg_criteria = generate_kg_criteria_template.render(
            purposes=purposes, 
            an_example=an_example, 
            more_examples=more_examples, 
            entity_types=entity_types, 
            entity_list=entity_list, 
            relation_list=relation_list)
        
        # Filter_entity template rendering
        filter_entity_criteria_string = json.dumps(criteria_json['Filter_entity'], indent=4, ensure_ascii=False)
        filter_entity_criteria_template = env.from_string(filter_entity_criteria_string)
        rendered_filter_entity_criteria = filter_entity_criteria_template.render(entity_types=entity_types)

        # Deduplicate_and_prune_kg template rendering
        deduplicate_and_prune_kg_criteria_string = json.dumps(criteria_json['Deduplicate_and_Prune_kg'], indent=4, ensure_ascii=False)
        deduplicate_and_prune_kg_criteria_template = env.from_string(deduplicate_and_prune_kg_criteria_string)
        rendered_deduplicate_and_prune_kg_criteria = deduplicate_and_prune_kg_criteria_template.render(purposes=purposes)
    except KeyError as e:
        logger.error(f"Missing key in JSON data: {e}")
        return

    generate_kg_criteria = json.loads(rendered_generate_kg_criteria)
    filter_entity_criteria = json.loads(rendered_filter_entity_criteria)
    deduplicate_and_prune_kg_criteria = json.loads(rendered_deduplicate_and_prune_kg_criteria)
    evaluate_kg_criteria = criteria_json['Evaluate_kg']

    # print("----------generate_kg_criteria----------")
    # print(generate_kg_criteria)
    # print("----------deduplicate_and_prune_kg_criteria----------")
    # print(deduplicate_and_prune_kg_criteria)
    # print("----------evaluate_kg_criteria----------")
    # print(evaluate_kg_criteria)    

    return generate_kg_criteria, filter_entity_criteria, deduplicate_and_prune_kg_criteria, evaluate_kg_criteria

if __name__ == "__main__":
    jinjaReadCriteria(dataset='base')