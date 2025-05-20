import argparse
import torch
import numpy as np
import random
import os
import logging
import datetime
import time
import gc
import json
from collections import defaultdict

from llm_utils import eval_match, parse_triples_from_llm, parse_triples_from_file
from llm_factory import get_local_llm_class, get_api_function
from getKeywords import get_indexed_keywords
from evalSimilar import get_k_similar_entities_dict_w_sapbert
from entrezFetcher import EntrezFetcher
from chunkRetriever import ChunkRetriever
from articleReader import ArticleReader
from metric_match import setup_wandb, track_match
from metric_recall_and_precision import RecallPrecisionCalculator, MetricLogger, ComprehensiveMetricLogger
from metric_all import plot_all_models_metrics, plot_all_models_triple_recall, detect_models_and_evaluators


current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
gc.set_threshold(300, 5, 5)  # set garbage collection threshold
api_models = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'gemini']
# local_models = ['phi-3.5-mini', 'mistral-7b', 'llama-3-8b', 'llama-3-70b', 'llama-3.1-405b']
local_models = ['phi-3.5-mini', 'mistral-7b', 'llama-3-8b', 'llama-3-70b']


def parse_args():
    parser = argparse.ArgumentParser(description="Constructing Knowledge Graph (KG) from medical literature.", usage='main.py [<args>] [-h | --help]')
    parser.add_argument('--model', type=str, choices=['phi-3.5-mini', 'mistral-7b', 'llama-3-8b', 'llama-3-70b', 'llama-3.1-405b', 'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'gemini'], default='mistral-7b', help='The language model for generating KG triples.')
    parser.add_argument('--evaluator', nargs='+', type=str, choices=['phi-3.5-mini', 'mistral-7b', 'llama-3-8b', 'llama-3-70b', 'llama-3.1-405b', 'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'gemini'], default=['mistral-7b'], help='The evaluator model(s) for evaluating the generated KG triples.')
    parser.add_argument('--keywords_expander', type=str, choices=['none', 'base', 'phi-3.5-mini', 'mistral-7b', 'llama-3-8b', 'llama-3-70b', 'llama-3.1-405b', 'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'gemini'], default='none', help='The expander for query keywords.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--mode', type=str, choices=['generate_and_eval', 'generate', 'eval', 'metric', 'mask', 'eval_exact_equiv', 'plot_all', 'conflict_resolve'], default='generate_and_eval', help='Mode of operation: generate_and_eval, generate, eval, metric, mask, eval_exact_equiv, plot_all, or conflict_resolve.')
    parser.add_argument('--run_id', type=str, default=None, help='Unique identifier for the run, used to find saved files during evaluation.')
    parser.add_argument('--enable', nargs='+', type=str, choices=['none', 'denoising_doc', 'criteria_more', 'criteria_graphjudger', 'criteria_graphusion', 'correctness_got', 'conflict_resolve', 'conflict_graphusion'], default=['none'], help='List of generation or expansion strategies to enable.')

    # setting for saved files
    parser.add_argument('--dataset', type=str, default='', help='Present KG dataset file name.')
    parser.add_argument('--keywords_file', type=str, default='', help='Keywords file name.')
    parser.add_argument('--save_dir', type=str, default='./output', help='Output file path to save the generated KG triples.')
    parser.add_argument('--log_file', type=str, default='logs', help='Log file name.')

    # setting for fetching, reading, and retrieving
    parser.add_argument('--article_acquisition_mode', type=str, choices=['fetch', 'read', 'copy'], default='fetch', help="Method to obtain articles: 'fetch' uses entrezFetcher, 'read' reads from an existing file.")
    parser.add_argument('--read_articles_file', type=str, choices=['none', 'pubmed.json', 'stackexchange.json'], default='none', help="Articles file name from which articles should be read if article_acquisition_mode is 'read'.")
    parser.add_argument('--articles_file', type=str, default='articles.json', help="Articles file name for which articles are fetched into when article_acquisition_mode is 'fetch'.")
    parser.add_argument('--chunks_file', type=str, default='chunks.json', help='Chunks file name.')
    parser.add_argument('--copy_chunks_file', type=str, help="Path to the generated_data.json file to copy chunks from when article_acquisition_mode is 'copy'.")
    parser.add_argument('--top_k', type=int, default=3, help='Number of documents to retrieve each time.')
    parser.add_argument('--retrieval_method', type=str, choices=['bm25s', 'rankbm25', 'bge'], default='bm25s', help='The method to use for retrieving relevant text chunks.')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations to repeat the KG generation process.')
    parser.add_argument('--start', type=int, default=0, help='Start iteration for generating or evaluating, overriding any saved progress.')

    # setting for entrez usage
    parser.add_argument('--email', type=str, default='anonymous', help='Email address for Entrez usage.')
    parser.add_argument('--api_key', type=str, default='anonymous', help='API key for Entrez usage.')
    parser.add_argument('--retmax', type=int, default=10, help='Number of articles to fetch each time.')
    parser.add_argument('--dbs', nargs='+', type=str, default=['pubmed'], choices=["pubmed", "protein", "nuccore", "ipg", "nucleotide", "structure", "genome", 
                        "annotinfo", "assembly", "bioproject", "biosample", "blastdbinfo", "books", "cdd", "clinvar", "gap", "gapplus", "grasp", "dbvar", "gene", 
                        "gds", "geoprofiles", "medgen", "mesh", "nlmcatalog", "omim", "orgtrack", "pmc", "popset", "proteinclusters", "pcassay", "protfam", "pccompound", 
                        "pcsubstance", "seqannot", "snp", "sra", "taxonomy", "biocollections", "gtr"], help='List of databases to query in Entrez.')
    
    # setting for evaluation
    parser.add_argument('--strictness', type=str, choices=['strict', 'lenient'], default='strict', help='The strictness of the similarity evaluation. strict means both head and tail must be similar, lenient means only one of them must be similar.')

    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = current_time

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    log_file = os.path.join(save_directory, f'{current_time}_{args.log_file}.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def convert_dict_keys_to_str(data):
    """Recursively convert dictionary keys to strings."""
    if isinstance(data, dict):
        return {str(key): convert_dict_keys_to_str(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_dict_keys_to_str(element) for element in data]
    else:
        return data

def convert_dict_keys_to_original_type(data):
    """Recursively convert dictionary keys from strings back to original types (e.g., tuples)."""
    if isinstance(data, dict):
        return {eval(key) if key.startswith('(') and key.endswith(')') else key: convert_dict_keys_to_original_type(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_dict_keys_to_original_type(element) for element in data]
    else:
        return data

def save_generated_data(args, generated_data):
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_path = os.path.join(save_directory, 'generated_data.json')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)
    logging.info(f"Generated data saved to {file_path}")

def save_evaluation_results(args, evaluator_name, llm_evals_per_iteration):
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_path = os.path.join(save_directory, f'{evaluator_name}_evaluation_results.json')

    # Convert keys to strings
    llm_evals_per_iteration_str_keys = convert_dict_keys_to_str(llm_evals_per_iteration)

    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(llm_evals_per_iteration_str_keys, f, ensure_ascii=False, indent=4)
    # logging.info(f"Evaluation results for evaluator {evaluator_name} saved to {file_path}")

    # 先写临时文件，写成功后再原子性覆盖
    with open(file_path + ".tmp", 'w', encoding='utf-8') as f:
        json.dump(llm_evals_per_iteration_str_keys, f, ensure_ascii=False, indent=4)
    os.replace(file_path + ".tmp", file_path)
    logging.info(f"Evaluation results for evaluator {evaluator_name} saved to {file_path}")

def save_triples(args, triples_list, file_name):
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_path = os.path.join(save_directory, file_name)
    
    seen_triples = set()  # create an empty set to store seen triples
    with open(file_path, 'w', encoding='utf-8') as file:
        for triple in triples_list:
            formatted_triple = ', '.join(triple)  # format the triple as a comma-separated string
            if formatted_triple not in seen_triples:  #  check if the formatted triple has been seen before
                file.write(formatted_triple + '\n')  # write the triple to the file and add a newline character after each triple
                seen_triples.add(formatted_triple)  # add the formatted triple to the set of seen triples

    return file_path, len(seen_triples)  # return the file path and the number of unique triples written to the file

def load_keywords(args):
    keywords = set()
    file_path = os.path.join('./data', args.dataset, args.keywords_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            keywords.add(line.strip())
    return keywords

def load_generated_data(args):
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    file_path = os.path.join(save_directory, 'generated_data.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)

    for data in generated_data:     # Convert triple lists back to tuples
        data['kg_triples'] = [tuple(triple) for triple in data['kg_triples']]
        data['matched_triples'] = [tuple(triple) for triple in data['matched_triples']]
        data['not_matched_triples'] = [tuple(triple) for triple in data['not_matched_triples']]

    logging.info(f"Generated data loaded from {file_path}")
    return generated_data

def load_evaluation_results(args, evaluator_name=None):
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    file_path = os.path.join(save_directory, f'{evaluator_name}_evaluation_results.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        evaluation_results_str_keys = json.load(f)

    # Convert keys back to original types
    evaluation_results = convert_dict_keys_to_original_type(evaluation_results_str_keys)

    logging.info(f"Evaluation results loaded from {file_path}")
    return evaluation_results

def load_last_iteration(args, current_mode=None, evaluator_name=None):
    if current_mode == "generate":
        try:
            saved_data = load_generated_data(args)
            last_iteration = max([data["iteration"] for data in saved_data])
            return last_iteration, saved_data
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"Failed to load generated data: {e}")
            return -1, []

    elif current_mode == "eval":
        try:
            saved_data = load_evaluation_results(args, evaluator_name)
            last_iteration = max([item["iteration"] for item in saved_data])
            return last_iteration, saved_data
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"Failed to load evaluation results for {evaluator_name}: {e}")
            return -1, []

def load_missing_iterations(args, generated_data=None, evaluation_results=None):
    """
    检测所有缺失的 iteration，返回需要评估的 iteration 列表和完成的 evaluation 数据。
    """
    if generated_data is None:
        generated_data = load_generated_data(args)

    all_iterations = {data['iteration'] for data in generated_data}  
    completed_iterations = set()  

    if evaluation_results:
        for evaluator_name, results in evaluation_results.items():
            if isinstance(results, list):  
                completed_iterations.update({item['iteration'] for item in results if 'iteration' in item})
            else:
                logging.warning(f"Unexpected format for evaluation results of {evaluator_name}: {type(results)}")

    missing_iterations = sorted(list(all_iterations - completed_iterations))  
    return missing_iterations, completed_iterations

def load_combined_data(args):
    """Load both generated data and evaluation results."""
    generated_data = load_generated_data(args)

    evaluation_results = {}
    for evaluator_name in args.evaluator:
        try:
            evaluation_results[evaluator_name] = load_evaluation_results(args, evaluator_name=evaluator_name)
        except FileNotFoundError:
            logging.warning(f"Evaluation results for evaluator {evaluator_name} not found.")
            evaluation_results[evaluator_name] = []

    return generated_data, evaluation_results

def filter_relevant_kg(args):
    """
    从 origin KG 中过滤出仅包含关键词相关的 triples、entities 和 relations。
    """
    gt_triples = parse_triples_from_file(dataset=args.dataset, file_name='triples.txt')
    keywords = load_keywords(args)
    logging.info(f"All keywords:\n{keywords}")

    filtered_triples = set()
    filtered_entities = set()
    filtered_relations = set()

    for triple in gt_triples:
        head, relation, tail = triple
        if head in keywords or tail in keywords or relation in keywords:
            filtered_triples.add(triple)
            filtered_entities.update([head, tail])
            filtered_relations.add(relation)
            
    del gt_triples
    gc.collect()
    torch.cuda.empty_cache()

    return filtered_triples, filtered_entities, filtered_relations
    
def find_and_collect_values(dictionary, keys_list):
    collected_values = []
    seen_values = set()

    for key in map(tuple, keys_list):
        if key in dictionary:  
            value_list = dictionary[key]
            for value in value_list:  
                if tuple(value) not in seen_values:  
                    collected_values.append(tuple(value))
                    seen_values.add(tuple(value))

    return collected_values

def generate(args):
    # Initialize wandb
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    wandb_run_id = setup_wandb(save_directory=save_directory)
    logging.info(f"WandB setup complete with run ID: {wandb_run_id}")

    # all_keywords = load_keywords(args)

    # Initialize recall_precision_calculator 
    relevant_triples, relevant_entities, relevant_relations = filter_relevant_kg(args)
    # logging.info(f"Relevant triples: datatype: {type(relevant_triples)}, content:\n{relevant_triples}")
    # logging.info(f"Relevant entities: datatype: {type(relevant_entities)}, content:\n{relevant_entities}")
    # logging.info(f"Relevant relations: datatype: {type(relevant_relations)}, content:\n{relevant_relations}")
    logging.info(f"Relevant triples count: {len(relevant_triples)}")
    logging.info(f"Relevant entities count: {len(relevant_entities)}")
    logging.info(f"Relevant relations count: {len(relevant_relations)}")
    recall_precision_calculator = RecallPrecisionCalculator(relevant_triples, relevant_entities, relevant_relations, mode=args.mode)

    # Initialize metric_logger
    metric_logger = MetricLogger(save_path=save_directory, generator_name=args.model, save_interval=1)

    # Initialize generator
    if args.model in api_models:
        # For API models, get functions directly
        generate_kg, _, _, _, _, _, _ = get_api_function(args.model)
        generator = None  # No generator instance
    else:
        # For local models, get the generator class and create an instance
        GeneratorClass = get_local_llm_class(args.model)
        generator = GeneratorClass()

    # Initialize fetcher, reader and retriever
    if args.article_acquisition_mode == 'fetch':
        fetcher = EntrezFetcher(articles_file=args.articles_file, current_time=args.run_id, keywords_file_without_extension=keywords_file_without_extension, log_file=args.log_file, dataset=args.dataset, save_dir=args.save_dir, email=args.email, api_key=args.api_key)
        reader = None
    elif args.article_acquisition_mode == 'read':
        fetcher = None
        reader = ArticleReader()
    elif args.article_acquisition_mode == 'copy':
        fetcher = None
        reader = None

    retriever = ChunkRetriever(retrieval_method=args.retrieval_method, article_acquisition_mode=args.article_acquisition_mode, articles_file=args.articles_file, read_articles_file=args.read_articles_file, chunks_file=args.chunks_file, current_time=args.run_id, keywords_file_without_extension=keywords_file_without_extension, log_file=args.log_file, dataset=args.dataset, save_dir=args.save_dir)

    # If using 'copy' mode, load the copy_chunks_file once at the beginning
    if args.article_acquisition_mode == 'copy':
        logging.info(f"[INIT] Preloading all chunks from {args.copy_chunks_file}")
        try:
            with open(args.copy_chunks_file, 'r', encoding='utf-8') as f:
                preloaded_copy_data = json.load(f)
            logging.info(f"[INIT] Loaded {len(preloaded_copy_data)} entries from copy_chunks_file.")
        except Exception as e:
            logging.error(f"[INIT] Failed to load copy_chunks_file: {e}")
            return  # Exit generation early
    else:
        preloaded_copy_data = None
        
    # Load last iteration if it exists
    last_iteration, generated_data = load_last_iteration(args, current_mode="generate")
    completed_iterations = {data['iteration'] for data in generated_data} if generated_data else set()
    logging.info(f"Completed iterations for generation: {completed_iterations}")
    if last_iteration == -1:  
        accumulated_triples = []
        accumulated_matched_triples = []
        generated_data = []  # list to store data needed for evaluation
    else:    # Convert lists back to tuples
        accumulated_triples = [triple for data in generated_data for triple in data["kg_triples"]]
        accumulated_matched_triples = [triple for data in generated_data for triple in data["matched_triples"]]
        logging.info(f"Resuming generation from iteration {last_iteration+1}...")

    start_iteration = args.start if args.start != 0 else (last_iteration + 1)

    for i in range(start_iteration, args.iterations):
        if i in completed_iterations:
            logging.info(f"Iteration {i} already completed. Skipping...")
            continue

        logging.info(f"--- Starting iteration {i}/{args.iterations - 1} ---")
        # Step 1: Extract keywords
        logging.info("[STEP 1] Extracting keywords.")
        keywords = get_indexed_keywords(dataset=args.dataset, input_file=args.keywords_file, i=i)
        logging.info(f"[STEP 1] Extracted keywords for iteration {i}:\n{keywords}")

        # Step 2: Fetch or read articles
        if args.article_acquisition_mode == 'fetch':
            logging.info("[STEP 2] Fetching articles based on keywords.")
            articles = fetcher(keywords=keywords, dbs=args.dbs, retmax=args.retmax)
            # Check that there is at least one non-empty 'Abstract' column entry
            if not articles.empty and articles['Abstract'].str.strip().str.len().gt(0).any():
                logging.info(f"[STEP 2] {args.retmax} related articles fetched.")
            else:
                logging.error(f"[STEP 2] No articles with non-empty abstracts fetched using the current keywords.")
                continue
            del articles
            gc.collect()
            torch.cuda.empty_cache()
        elif args.article_acquisition_mode == 'read':
            logging.info(f"[STEP 2] Reading articles in {args.read_articles_file}, one random sample is shown below:")
            reader(read_articles_file=args.read_articles_file)
        elif args.article_acquisition_mode == 'copy':
            logging.info(f"[STEP 2-3] Copying chunks from preloaded copy_chunks_file.")
            if i < len(preloaded_copy_data):
                chunks = preloaded_copy_data[i].get('chunks', [])
                if not chunks:
                    logging.warning(f"[STEP 2-3] No chunks found for iteration {i}; skipping...")
                    continue
                # logging.info(f"[STEP 2-3] Copied {len(chunks)} chunks for iteration {i}.")
            else:
                logging.warning(f"[STEP 2-3] Not enough entries in preloaded copy_chunks_file for iteration {i}; skipping...")
                continue

        # Step 3: Retrieve sentences/chunks from above articles
        if args.article_acquisition_mode != 'copy':
            logging.info("[STEP 3] Retrieving chunks from articles.")
            chunks = retriever(keywords=keywords, top_k=args.top_k)
            logging.info(f"[STEP 3] {args.top_k} chunks extracted:\n{chunks}")
            if len(chunks) == 0:
                logging.info(f"[STEP 3] No chunks retrieved for iteration {i}; skipping to next iteration.")
                continue

        # Step 4: Generate KG triples
        logging.info("[STEP 4] Generating KG triples.")
        if args.model in api_models:
            kg_string = generate_kg(args.keywords_expander, args.dataset, keywords, chunks, args.enable)
        else:
            if generator is None:
                logging.info(f"Separately loading generator model {args.model} because either generator or evaluator is large.")
                generator = GeneratorClass()
            kg_string = generator.generate_kg(args.keywords_expander, args.dataset, keywords, chunks, args.enable)

        logging.info(f"[STEP 4] Generated KG strings has {len(kg_string)} words.")
        kg_triples = parse_triples_from_llm(kg_string)
        del kg_string
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"[STEP 4] Generated KG triples in iteration {i}:\n{kg_triples}")
        count_last_iter = len(accumulated_triples)
        accumulated_triples.extend(kg_triples)
        count = len(accumulated_triples)
        count_new = count - count_last_iter
        logging.info(f"[STEP 4] Generated {len(kg_triples)} KG triples, among {count_new} are new triples. Accumulated {count} KG triples till now.")

        # Step 5: Eval match metric
        logging.info("[STEP 5] Evaluating the generated triples with match metric compared to the present KG.")
        matched_triples, not_matched_triples = eval_match(kg_triples, args.dataset)
        accumulated_matched_triples.extend(matched_triples)
        logging.info(f"[STEP 5] Generated {len(matched_triples)} matched KG triples:\n{matched_triples}")
        logging.info(f"[STEP 5] Generated {len(not_matched_triples)} not matched KG triples:\n{not_matched_triples}")

        # Step 6: Analyze k-similar entities using SapBERT (This calculation might also need large memory, but the result does not need large memory)
        logging.info("[STEP 6] Analyzing the generated entities with SapBERT-based method compared to the present KG. Note that this step does not consider similarity in relation.")
        # if args.model == 'llama-3-70b':    # try uncomment this block if you got OOM, but this will slow down the process
        #     del generator
        #     torch.cuda.empty_cache()
        #     gc.collect()
        #     generator = None
        #     logging.info(f"Unloading generator model {args.model} to free up memory for SapBERT-based top-k similar entities calculation.")
        k_similar_entities_dict = get_k_similar_entities_dict_w_sapbert(not_matched_triples, args.dataset, k=6)
        logging.info(f"[STEP 6] Calculated top-k similar entities for not matched KG triples using SapBERT-based method:\n{k_similar_entities_dict}")

        # Store data needed for evaluation
        generated_data.append({
            'iteration': i,
            'keywords': keywords,
            'chunks': chunks,
            'kg_triples': kg_triples,
            'matched_triples': matched_triples,
            'not_matched_triples': not_matched_triples,
            'k_similar_entities_dict': k_similar_entities_dict,
        })

        if (i + 1) % 1 == 0:
            logging.info(f"--- Saving intermediate data at iteration {i} ---")
            save_generated_data(args, generated_data)

        # Track match after x iteration
        track_match(
            iteration=i,
            kg_triples=accumulated_triples,
            matched_triples=accumulated_matched_triples,
            save_image_interval=5,
            save_dir=save_directory
        )
        
        # calculate recall every full iterations of keywords file
        # if (i + 1) % len(all_keywords) == 0:
        # logging.info(f"Calculating recall after {i + 1} iterations...")
        recall_and_precision = recall_precision_calculator.calculate_recall_and_precision(accumulated_generated_triples=accumulated_triples, description="match*")
        metric_logger.log_metrics(iteration=i, metrics=recall_and_precision)

        del keywords, chunks, kg_triples
        del matched_triples, not_matched_triples, k_similar_entities_dict
        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"--- Saving final KG triples ---")
    accumulated_triples_path, num_all = save_triples(args, accumulated_triples, 'accumulated_all.txt')
    accumulated_matched_triples_path, num_match = save_triples(args, accumulated_matched_triples, 'accumulated_matched.txt')
    logging.info(f"{num_all} Accumulated KG triples saved to {accumulated_triples_path}")
    logging.info(f"{num_match} Accumulated matched KG triples saved to {accumulated_matched_triples_path}")

    save_generated_data(args, generated_data)

    if generator is not None:
        generator.__del__()
        del generator
        generator = None
        torch.cuda.empty_cache()
        gc.collect()

def evaluate(args, generated_data=None):
    if generated_data is None:
        generated_data = load_generated_data(args)

    accumulated_kg_triples = []
    for data in generated_data:
        accumulated_kg_triples.extend(data['kg_triples'])
    accumulated_kg_triples = list(set(accumulated_kg_triples))  
    logging.info(f"Loaded {len(accumulated_kg_triples)} accumulated KG triples from all iterations")

    evaluation_results = {}
    for evaluator_name in args.evaluator:
        try:
            evaluation_results[evaluator_name] = load_evaluation_results(args, evaluator_name=evaluator_name)
        except FileNotFoundError:
            logging.warning(f"Evaluation results for {evaluator_name} not found.")
            evaluation_results[evaluator_name] = []
    missing_iterations, completed_iterations = load_missing_iterations(args, generated_data, evaluation_results)
    logging.info(f"Completed iterations: {sorted(completed_iterations)}")
    logging.info(f"Missing iterations to evaluate: {sorted(missing_iterations)}")

    if args.start is not None and args.start > 0:
        missing_iterations = [i for i in missing_iterations if i >= args.start]
        logging.info(f"Filtered missing iterations starting from {args.start}: {sorted(missing_iterations)}")

    if not missing_iterations:
        logging.info("No missing iterations to evaluate. Exiting evaluation.")
        return
    
    for evaluator_name in args.evaluator:
        # Initialize wandb
        keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
        save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
        wandb_run_id = setup_wandb(save_directory=save_directory, evaluator_name=evaluator_name)
        logging.info(f"WandB setup complete with run ID: {wandb_run_id}")

        # Initialize recall_precision_calculator 
        relevant_triples, relevant_entities, relevant_relations = filter_relevant_kg(args)
        recall_precision_calculator = RecallPrecisionCalculator(relevant_triples, relevant_entities, relevant_relations, mode=args.mode)

        # Initialize metric_logger
        metric_logger = MetricLogger(save_path=save_directory, generator_name=args.model, evaluator_name=evaluator_name, save_interval=1)

        # # Load existing evaluation results if available
        # last_iteration, llm_evals_per_iteration = load_last_iteration(args, current_mode="eval", evaluator_name=evaluator_name)
        # completed_iterations = {data['iteration'] for data in llm_evals_per_iteration} if llm_evals_per_iteration else set()
        # logging.info(f"Completed iterations for evaluator {evaluator_name}: {completed_iterations}")

        # if last_iteration == -1:  
        #     llm_evals_per_iteration = []
        # else:
        #     logging.info(f"Resuming evaluation with {evaluator_name} model from iteration {last_iteration+1}...")

        # if args.start != 0:
        #     start_iteration = args.start
        # else:
        #     start_iteration = last_iteration + 1

        if evaluator_name in evaluation_results:
            llm_evals_per_iteration = evaluation_results[evaluator_name]
        else:
            llm_evals_per_iteration = []

        llm_evals_per_evaluator = {
            'accumulated_not_match_but_similar_triples_w_promptlink': [],
            'accumulated_corresponding_similar_gt_triples_w_promptlink': [],
            'accumulated_not_match_but_equiv_triples': [],
            'accumulated_corresponding_equiv_gt_triples': [],
            'accumulated_not_match_but_exact_equiv_triples': [],
            'accumulated_corresponding_exact_equiv_gt_triples': [],
            'accumulated_not_match_but_correct': [],
            'accumulated_not_match_and_wrong': [],
            'accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink': [],  # for analysis
            'accumulated_corresponding_equiv_and_correct_gt_triples': [],
            'accumulated_corresponding_exact_equiv_and_correct_gt_triples': [],    # for medical domain
            'accumulated_not_match_not_equiv_but_correct': [],
            'accumulated_not_match_not_exact_equiv_but_correct': [],
            'accumulated_detected_conflict_triples_on_similar': [],
            'accumulated_removal_conflict_triples_on_similar': [],
            # 'accumulated_detected_conflict_triples_on_equiv': [],
            # 'accumulated_removal_conflict_triples_on_equiv': [],
            # 'accumulated_detected_conflict_triples_on_exact_equiv': [],
            # 'accumulated_removal_conflict_triples_on_exact_equiv': [],
            'accumulated_not_match_not_exact_equiv_but_correct_no_conflict': [],
            'accumulated_gt_triples_to_remove': [],
        }
        # Load existing evaluation results for each evaluator if available
        for iteration_eval in llm_evals_per_iteration:
            llm_evals_per_evaluator['accumulated_not_match_but_similar_triples_w_promptlink'].extend(list(iteration_eval.get('not_match_but_similar_triples_w_promptlink', {}).keys()))
            llm_evals_per_evaluator['accumulated_corresponding_similar_gt_triples_w_promptlink'].extend(iteration_eval.get('corresponding_similar_gt_triples_w_promptlink', []))
            llm_evals_per_evaluator['accumulated_not_match_but_equiv_triples'].extend(list(iteration_eval.get('not_match_but_equiv_triples', {}).keys()))
            llm_evals_per_evaluator['accumulated_corresponding_equiv_gt_triples'].extend(iteration_eval.get('corresponding_equiv_gt_triples', []))
            llm_evals_per_evaluator['accumulated_not_match_but_exact_equiv_triples'].extend(list(iteration_eval.get('not_match_but_exact_equiv_triples', {}).keys()))
            llm_evals_per_evaluator['accumulated_corresponding_exact_equiv_gt_triples'].extend(iteration_eval.get('corresponding_exact_equiv_gt_triples', []))
            llm_evals_per_evaluator['accumulated_not_match_but_correct'].extend(iteration_eval.get('not_match_but_correct', []))
            llm_evals_per_evaluator['accumulated_not_match_and_wrong'].extend(iteration_eval.get('not_match_and_wrong', []))
            llm_evals_per_evaluator['accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink'].extend(iteration_eval.get('corresponding_similar_and_correct_gt_triples_w_promptlink', []))
            llm_evals_per_evaluator['accumulated_corresponding_equiv_and_correct_gt_triples'].extend(iteration_eval.get('corresponding_equiv_and_correct_gt_triples', []))
            llm_evals_per_evaluator['accumulated_corresponding_exact_equiv_and_correct_gt_triples'].extend(iteration_eval.get('corresponding_exact_equiv_and_correct_gt_triples', []))
            llm_evals_per_evaluator['accumulated_not_match_not_equiv_but_correct'].extend(iteration_eval.get('not_match_not_equiv_but_correct', []))
            llm_evals_per_evaluator['accumulated_not_match_not_exact_equiv_but_correct'].extend(iteration_eval.get('not_match_not_exact_equiv_but_correct', []))
            llm_evals_per_evaluator['accumulated_detected_conflict_triples_on_similar'].extend(list(iteration_eval.get('detected_conflict_triples_on_similar', {}).keys()))
            llm_evals_per_evaluator['accumulated_removal_conflict_triples_on_similar'].extend(list(iteration_eval.get('removal_conflict_triples_on_similar', {}).keys()))
            # llm_evals_per_evaluator['accumulated_detected_conflict_triples_on_equiv'].extend(list(iteration_eval.get('detected_conflict_triples_on_equiv', {}).keys()))
            # llm_evals_per_evaluator['accumulated_removal_conflict_triples_on_equiv'].extend(list(iteration_eval.get('removal_conflict_triples_on_equiv', {}).keys()))
            # llm_evals_per_evaluator['accumulated_detected_conflict_triples_on_exact_equiv'].extend(list(iteration_eval.get('detected_conflict_triples_on_exact_equiv', {}).keys()))
            # llm_evals_per_evaluator['accumulated_removal_conflict_triples_on_exact_equiv'].extend(list(iteration_eval.get('removal_conflict_triples_on_exact_equiv', {}).keys()))
            llm_evals_per_evaluator['accumulated_not_match_not_exact_equiv_but_correct_no_conflict'].extend(iteration_eval.get('not_match_not_exact_equiv_but_correct_no_conflict', []))
            llm_evals_per_evaluator['accumulated_gt_triples_to_remove'].extend(iteration_eval.get('gt_triples_to_remove', []))
            
        # Initialize the evaluator for this iteration
        if evaluator_name in api_models:    # For API evaluators, get functions directly
            _, get_similar_triples_dict_w_promptlink, eval_similar_w_promptlink, eval_accuracy, eval_equivalence, eval_exact_equiv, eval_confliction = get_api_function(evaluator_name)
        else:     
            EvaluatorClass = get_local_llm_class(evaluator_name)
            evaluator_instance = EvaluatorClass()
            get_similar_triples_dict_w_promptlink = evaluator_instance.get_similar_triples_dict_w_promptlink
            eval_similar_w_promptlink = evaluator_instance.eval_similar_w_promptlink
            eval_accuracy = evaluator_instance.eval_accuracy
            eval_equivalence = evaluator_instance.eval_equivalence
            eval_exact_equiv = evaluator_instance.eval_exact_equiv
            eval_confliction = evaluator_instance.eval_confliction

        for data in generated_data:
            # i = data['iteration']
            # if i < start_iteration:
            #     continue  
            # if i in completed_iterations:
            #     logging.info(f"Iteration {i} already completed for evaluator {evaluator_name}. Skipping...")
            #     continue

            i = data['iteration']
            if i not in missing_iterations:
                continue  
            logging.info(f"--- Starting evaluation for iteration {i} using {evaluator_name} ---")
            
            not_matched_triples = data['not_matched_triples']
            k_similar_entities_dict = data['k_similar_entities_dict']
            chunks = data['chunks']

            # Record GPU model (if available)
            if torch.cuda.is_available():
                gpu_model = torch.cuda.get_device_name(0)
            else:
                gpu_model = "CPU"

            # Start timing
            start_time = time.time()

            # Do evaluation for the current evaluator model
            logging.info(f"--- Iteration {i}/{args.iterations - 1}: Evaluating with evaluator model: {evaluator_name} ---")
            
            # Step 6: Analyze similar triples using PromptLink
            logging.info("[STEP 6] Analyzing the generated entities with PromptLink-based method compared to the present KG.")
            similar_entities_dict, similar_gt_triples_dict = get_similar_triples_dict_w_promptlink(not_matched_triples, k_similar_entities_dict, args.dataset, strictness=args.strictness, keywords_expander=args.keywords_expander)
            logging.info(f"[STEP 6] Confirmed similar entities for not matched KG triples using PromptLink-based method:\n{similar_entities_dict}")
            logging.info(f"[STEP 6] Found similar present KG triples for {len(similar_gt_triples_dict)} not matched KG triples:\n{similar_gt_triples_dict}")

            # Step 6: Eval similarity metric using PromptLink
            logging.info("[STEP 6] Evaluating the generated triples with PromptLink-based similarity metric compared to the present KG.")
            similar_gt_triples_dict_w_promptlink = eval_similar_w_promptlink(args.keywords_expander, similar_gt_triples_dict, args.dataset)
            corresponding_similar_gt_triples_w_promptlink = find_and_collect_values(similar_gt_triples_dict_w_promptlink, list(similar_gt_triples_dict_w_promptlink))
            llm_evals_per_evaluator['accumulated_not_match_but_similar_triples_w_promptlink'].extend(similar_gt_triples_dict_w_promptlink.keys())
            llm_evals_per_evaluator['accumulated_corresponding_similar_gt_triples_w_promptlink'].extend(corresponding_similar_gt_triples_w_promptlink)
            logging.info(f"[STEP 6] Confirmed similar present KG triples for {len(similar_gt_triples_dict_w_promptlink)} not matched KG triples using PromptLink-based similarity metric:\n{similar_gt_triples_dict_w_promptlink}")
            logging.info(f"[STEP 6] Confirmed {len(corresponding_similar_gt_triples_w_promptlink)} corresponding similar present KG triples:\n{corresponding_similar_gt_triples_w_promptlink}")
            
            # Step 7: Eval equivalence metric
            logging.info("[STEP 7] Evaluating the generated triples with equivalence metric compared to the present KG.")
            equivalent_gt_triples_dict = eval_equivalence(similar_gt_triples_dict_w_promptlink)
            corresponding_equiv_gt_triples = find_and_collect_values(equivalent_gt_triples_dict, list(equivalent_gt_triples_dict))
            llm_evals_per_evaluator['accumulated_not_match_but_equiv_triples'].extend(equivalent_gt_triples_dict.keys())
            llm_evals_per_evaluator['accumulated_corresponding_equiv_gt_triples'].extend(corresponding_equiv_gt_triples)
            logging.info(f"[STEP 7] Confirmed equivalent present KG triples for {len(equivalent_gt_triples_dict)} not matched KG triples using equivalence metric:\n{equivalent_gt_triples_dict}")
            logging.info(f"[STEP 7] Confirmed {len(corresponding_equiv_gt_triples)} corresponding equivalent present KG triples:\n{corresponding_equiv_gt_triples}")

            # Step 8: Eval exact equivalence metric
            logging.info("[STEP 8] Evaluating the generated triples with exact equivalence metric compared to the present KG. (based on the results of STEP 7)")
            exact_equiv_gt_triples_dict = eval_exact_equiv(equivalent_gt_triples_dict)
            corresponding_exact_equiv_gt_triples = find_and_collect_values(exact_equiv_gt_triples_dict, list(exact_equiv_gt_triples_dict))
            llm_evals_per_evaluator['accumulated_not_match_but_exact_equiv_triples'].extend(exact_equiv_gt_triples_dict.keys())
            llm_evals_per_evaluator['accumulated_corresponding_exact_equiv_gt_triples'].extend(corresponding_exact_equiv_gt_triples)
            logging.info(f"[STEP 8] Confirmed exact equivalent present KG triples for {len(exact_equiv_gt_triples_dict)} not matched KG triples using exact equivalence metric:\n{exact_equiv_gt_triples_dict}")
            logging.info(f"[STEP 8] Confirmed {len(corresponding_exact_equiv_gt_triples)} corresponding equivalent present KG triples:\n{corresponding_exact_equiv_gt_triples}")

            # Step 9: Eval accuracy metric
            logging.info("[STEP 9] Evaluating the generated triples with accuracy metric.")
            not_match_but_correct, not_match_and_wrong = eval_accuracy(not_matched_triples, chunks, args.enable)
            corresponding_equiv_and_correct_gt_triples = find_and_collect_values(equivalent_gt_triples_dict, not_match_but_correct)
            corresponding_similar_and_correct_gt_triples_w_promptlink = find_and_collect_values(similar_gt_triples_dict_w_promptlink, not_match_but_correct)
            corresponding_exact_equiv_and_correct_gt_triples = find_and_collect_values(exact_equiv_gt_triples_dict, not_match_but_correct)
            llm_evals_per_evaluator['accumulated_not_match_but_correct'].extend(not_match_but_correct)
            llm_evals_per_evaluator['accumulated_not_match_and_wrong'].extend(not_match_and_wrong)
            llm_evals_per_evaluator['accumulated_corresponding_equiv_and_correct_gt_triples'].extend(corresponding_equiv_and_correct_gt_triples)
            llm_evals_per_evaluator['accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink'].extend(corresponding_similar_and_correct_gt_triples_w_promptlink)
            llm_evals_per_evaluator['accumulated_corresponding_exact_equiv_and_correct_gt_triples'].extend(corresponding_exact_equiv_and_correct_gt_triples)
            logging.info(f"[STEP 9] Confirmed {len(not_match_but_correct) } not match but correct KG triples:\n{not_match_but_correct}")
            logging.info(f"[STEP 9] Confirmed {len(not_match_and_wrong)} not match and wrong KG triples:\n{not_match_and_wrong}")
            logging.info(f"[STEP 9] Confirmed {len(corresponding_equiv_and_correct_gt_triples)} corresponding equivalent and correct present KG triples:\n{corresponding_equiv_and_correct_gt_triples}")
            logging.info(f"[STEP 9] Confirmed {len(corresponding_similar_and_correct_gt_triples_w_promptlink)} corresponding similar and correct present KG triples using PromptLink-based similarity metric:\n{corresponding_similar_and_correct_gt_triples_w_promptlink}")
            logging.info(f"[STEP 9] Confirmed {len(corresponding_exact_equiv_and_correct_gt_triples)} corresponding exactly equivalent and correct present KG triples:\n{corresponding_exact_equiv_and_correct_gt_triples}")

            # Step 10: Get expanded knowledge (new and correct knowledge triples)
            not_match_not_equiv_but_correct = [triple for triple in not_match_but_correct if triple not in list(equivalent_gt_triples_dict)]
            not_match_not_exact_equiv_but_correct = [triple for triple in not_match_but_correct if triple not in list(exact_equiv_gt_triples_dict)]
            llm_evals_per_evaluator['accumulated_not_match_not_equiv_but_correct'].extend(not_match_not_equiv_but_correct)
            llm_evals_per_evaluator['accumulated_not_match_not_exact_equiv_but_correct'].extend(not_match_not_exact_equiv_but_correct)
            logging.info(f"[STEP 10] Get {len(not_match_not_equiv_but_correct)} expanded (not match not equivalent but correct) KG triples):\n{not_match_not_equiv_but_correct}")
            logging.info(f"[STEP 10] Get {len(not_match_not_exact_equiv_but_correct)} expanded (not match not exactly equivalent but correct) KG triples):\n{not_match_not_exact_equiv_but_correct}")

            # Step 11: Conflict resolution, note that,
            # (1) Users only need to detect the conflicts in the expanded knowledge triples (e.g., the not match not exact equivalent but correct KG triples), detecting conflicts on the equivalent triples is for monitored purpose only. 
            # (2) Even conflicts are detected, the users can choose to remove the conflict triples or not, because conflicts does not mean incorrect triples, conflicts knowledge usually coexists.  
            detected_conflict_triple_dict_on_similar, removal_conflict_triple_dict_on_similar = eval_confliction(similar_gt_triples_dict_w_promptlink, chunks, args.enable)
            # detected_conflict_triple_dict_on_equiv, removal_conflict_triple_dict_on_equiv = eval_confliction(equivalent_gt_triples_dict, chunks, args.enable)
            # detected_conflict_triple_dict_on_exact_equiv, removal_conflict_triple_dict_on_exact_equiv = eval_confliction(exact_equiv_gt_triples_dict, chunks, args.enable)
            llm_evals_per_evaluator['accumulated_detected_conflict_triples_on_similar'].extend(detected_conflict_triple_dict_on_similar.keys())
            llm_evals_per_evaluator['accumulated_removal_conflict_triples_on_similar'].extend(removal_conflict_triple_dict_on_similar.keys())
            # llm_evals_per_evaluator['accumulated_detected_conflict_triples_on_equiv'].extend(detected_conflict_triple_dict_on_equiv.keys())
            # llm_evals_per_evaluator['accumulated_removal_conflict_triples_on_equiv'].extend(removal_conflict_triple_dict_on_equiv.keys())
            # llm_evals_per_evaluator['accumulated_detected_conflict_triples_on_exact_equiv'].extend(detected_conflict_triple_dict_on_exact_equiv.keys())
            # llm_evals_per_evaluator['accumulated_removal_conflict_triples_on_exact_equiv'].extend(removal_conflict_triple_dict_on_exact_equiv.keys())
            logging.info(f"[STEP 11] Detected {len(detected_conflict_triple_dict_on_similar)} conflict triples on similar KG triples:\n{detected_conflict_triple_dict_on_similar}")
            logging.info(f"[STEP 11] Removed {len(removal_conflict_triple_dict_on_similar)} conflict triples on similar KG triples:\n{removal_conflict_triple_dict_on_similar}")
            # logging.info(f"[STEP 11] Detected {len(detected_conflict_triple_dict_on_equiv)} conflict triples on equivalent KG triples:\n{detected_conflict_triple_dict_on_equiv}")
            # logging.info(f"[STEP 11] Removed {len(removal_conflict_triple_dict_on_equiv)} conflict triples on equivalent KG triples:\n{removal_conflict_triple_dict_on_equiv}")
            # logging.info(f"[STEP 11] Detected {len(detected_conflict_triple_dict_on_exact_equiv)} conflict triples on exactly equivalent KG triples:\n{detected_conflict_triple_dict_on_exact_equiv}")
            # logging.info(f"[STEP 11] Removed {len(removal_conflict_triple_dict_on_exact_equiv)} conflict triples on exactly equivalent KG triples:\n{removal_conflict_triple_dict_on_exact_equiv}")

            # Step 12: Get expanded knowledge with conflict resoved (new and correct and not conflict knowledge triples)
            removal_conflict_triples = {tuple(item) if not isinstance(item, tuple) else item for v in removal_conflict_triple_dict_on_similar.values() for item in v}
            not_match_not_exact_equiv_but_correct_no_conflict = [triple for triple in not_match_not_exact_equiv_but_correct if triple not in removal_conflict_triples]
            gt_triples_to_remove = [triple for triple in removal_conflict_triple_dict_on_similar.keys() if triple in removal_conflict_triples]
            llm_evals_per_evaluator['accumulated_not_match_not_exact_equiv_but_correct_no_conflict'].extend(not_match_not_exact_equiv_but_correct_no_conflict)
            llm_evals_per_evaluator['accumulated_gt_triples_to_remove'].extend(gt_triples_to_remove)
            logging.info(f"[STEP 12] Get {len(not_match_not_exact_equiv_but_correct_no_conflict)} expanded (not match not exactly equivalent but correct and no conflict) KG triples):\n{not_match_not_exact_equiv_but_correct_no_conflict}")
            logging.info(f"[STEP 12] Get {len(gt_triples_to_remove)} conflict triples to remove from originally exisiting KG triples:\n{gt_triples_to_remove}")

            # End timing
            end_time = time.time()
            inference_time = end_time - start_time

            # Collect evaluation results per iteration
            iteration_evals = {
                'iteration': i,
                'not_match_but_similar_entities_dict': similar_entities_dict,
                'not_match_but_similar_gt_triples_dict': similar_gt_triples_dict,
                'not_match_but_similar_triples_w_promptlink': similar_gt_triples_dict_w_promptlink,
                'corresponding_similar_gt_triples_w_promptlink': corresponding_similar_gt_triples_w_promptlink,
                'not_match_but_equiv_triples': equivalent_gt_triples_dict,
                'corresponding_equiv_gt_triples': corresponding_equiv_gt_triples,
                'not_match_but_exact_equiv_triples': exact_equiv_gt_triples_dict,
                'corresponding_exact_equiv_gt_triples': corresponding_exact_equiv_gt_triples,
                'not_match_but_correct': not_match_but_correct,
                'not_match_and_wrong': not_match_and_wrong,
                'corresponding_similar_and_correct_gt_triples_w_promptlink': corresponding_similar_and_correct_gt_triples_w_promptlink,
                'corresponding_equiv_and_correct_gt_triples': corresponding_equiv_and_correct_gt_triples,
                'corresponding_exact_equiv_and_correct_gt_triples': corresponding_exact_equiv_and_correct_gt_triples,
                'not_match_not_equiv_but_correct': not_match_not_equiv_but_correct,
                'not_match_not_exact_equiv_but_correct': not_match_not_exact_equiv_but_correct,
                'detected_conflict_triple_dict_on_similar': detected_conflict_triple_dict_on_similar,
                'removal_conflict_triple_dict_on_similar': removal_conflict_triple_dict_on_similar,
                # 'detected_conflict_triple_dict_on_equiv': detected_conflict_triple_dict_on_equiv,
                # 'removal_conflict_triple_dict_on_equiv': removal_conflict_triple_dict_on_equiv,                
                # 'detected_conflict_triple_dict_on_exact_equiv': detected_conflict_triple_dict_on_exact_equiv,
                # 'removal_conflict_triple_dict_on_exact_equiv': removal_conflict_triple_dict_on_exact_equiv,
                'not_match_not_exact_equiv_but_correct_no_conflict': not_match_not_exact_equiv_but_correct_no_conflict, 
                'gt_triples_to_remove': gt_triples_to_remove,
                'inference_time_seconds': inference_time,  # Add inference time
                'gpu_model': gpu_model,  # Add GPU model information
            }
            llm_evals_per_iteration.append(iteration_evals)

            if (i + 1) % 1 == 0:
                logging.info(f"--- Saving intermediate data at iteration {i} ---")
                save_evaluation_results(args, evaluator_name, llm_evals_per_iteration)

            recalls_and_precisions = recall_precision_calculator.calculate_all_recalls_and_precisions(
                accumulated_generated_triples=accumulated_kg_triples,
                accumulated_not_match_but_similar_triples_w_promptlink=llm_evals_per_evaluator['accumulated_not_match_but_similar_triples_w_promptlink'],
                accumulated_corresponding_similar_gt_triples_w_promptlink=llm_evals_per_evaluator['accumulated_corresponding_similar_gt_triples_w_promptlink'],
                accumulated_not_match_but_equiv_triples=llm_evals_per_evaluator['accumulated_not_match_but_equiv_triples'],
                accumulated_corresponding_equiv_gt_triples=llm_evals_per_evaluator['accumulated_corresponding_equiv_gt_triples'],
                accumulated_not_match_but_exact_equiv_triples=llm_evals_per_evaluator['accumulated_not_match_but_exact_equiv_triples'],
                accumulated_corresponding_exact_equiv_gt_triples=llm_evals_per_evaluator['accumulated_corresponding_exact_equiv_gt_triples'],
                accumulated_not_match_but_correct=llm_evals_per_evaluator['accumulated_not_match_but_correct'],
                accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink=llm_evals_per_evaluator['accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink'],
                accumulated_corresponding_equiv_and_correct_gt_triples=llm_evals_per_evaluator['accumulated_corresponding_equiv_and_correct_gt_triples'],
                accumulated_corresponding_exact_equiv_and_correct_gt_triples=llm_evals_per_evaluator['accumulated_corresponding_exact_equiv_and_correct_gt_triples']
            )
            metric_logger.log_metrics(iteration=i, metrics=recalls_and_precisions)

            del similar_entities_dict, similar_gt_triples_dict, similar_gt_triples_dict_w_promptlink, corresponding_similar_gt_triples_w_promptlink, 
            del equivalent_gt_triples_dict, corresponding_equiv_gt_triples, exact_equiv_gt_triples_dict, corresponding_exact_equiv_gt_triples 
            del not_match_but_correct, not_match_and_wrong
            del corresponding_similar_and_correct_gt_triples_w_promptlink, corresponding_equiv_and_correct_gt_triples, corresponding_exact_equiv_and_correct_gt_triples
            del not_match_not_equiv_but_correct, not_match_not_exact_equiv_but_correct
            del detected_conflict_triple_dict_on_similar, removal_conflict_triple_dict_on_similar
            # del detected_conflict_triple_dict_on_equiv, removal_conflict_triple_dict_on_equiv
            # del detected_conflict_triple_dict_on_exact_equiv, removal_conflict_triple_dict_on_exact_equiv
            del not_match_not_exact_equiv_but_correct_no_conflict, gt_triples_to_remove
            gc.collect()
            torch.cuda.empty_cache()

        if 'evaluator_instance' in locals() and evaluator_instance is not None:
            evaluator_instance.__del__()
            del evaluator_instance
            torch.cuda.empty_cache()
            gc.collect()

        logging.info(f"--- Saving final evaluation results for evaluator model: {evaluator_name} ---")
        save_evaluation_results(args, evaluator_name, llm_evals_per_iteration)
        del llm_evals_per_iteration
        gc.collect()
        torch.cuda.empty_cache()

        for key, value in llm_evals_per_evaluator.items():
            file_name = f'{evaluator_name}_{key}.txt'
            file_path, num_items = save_triples(args, value, file_name)
            logging.info(f"{num_items} triples saved to {file_path}")

        del llm_evals_per_evaluator 
        gc.collect()
        torch.cuda.empty_cache()

def evaluate_exact_equiv(args):
    """
    仅评估 `exact_equiv`，检查已有 `evaluation_results` 文件，并更新内容。
    """
    for evaluator_name in args.evaluator:
        try:
            evaluation_results = load_evaluation_results(args, evaluator_name=evaluator_name)
        except FileNotFoundError:
            logging.warning(f"Evaluation results for {evaluator_name} not found. Skipping evaluator.")
            continue
        logging.info(f"Loaded evaluation results for evaluator {evaluator_name}.")

        updated_evaluation_results = evaluation_results.copy()

        exact_equiv_missing = [data for data in evaluation_results if not all(key in data for key in [
            "not_match_but_exact_equiv_triples",
            "corresponding_exact_equiv_gt_triples",
            "corresponding_exact_equiv_and_correct_gt_triples",
            "not_match_not_exact_equiv_but_correct"])]

        if not exact_equiv_missing:
            logging.info(f"All iterations already contain `exact_equiv` evaluation for {evaluator_name}.")
            continue  

        if evaluator_name in api_models:
            _, _, _, _, _, eval_exact_equiv, _ = get_api_function(evaluator_name)
        else:
            EvaluatorClass = get_local_llm_class(evaluator_name)
            evaluator_instance = EvaluatorClass()
            eval_exact_equiv = evaluator_instance.eval_exact_equiv

        for eval_data in exact_equiv_missing:
            iteration = eval_data["iteration"]
            if iteration < args.start: 
                continue  
            logging.info(f"Evaluating `exact_equiv` for iteration {iteration} using {evaluator_name}.")

            equivalent_gt_triples_dict = eval_data.get("not_match_but_equiv_triples", {})
            exact_equiv_gt_triples_dict = eval_exact_equiv(equivalent_gt_triples_dict)

            eval_data["not_match_but_exact_equiv_triples"] = exact_equiv_gt_triples_dict
            eval_data["corresponding_exact_equiv_gt_triples"] = find_and_collect_values(exact_equiv_gt_triples_dict, list(exact_equiv_gt_triples_dict))

            not_match_but_correct = eval_data.get("not_match_but_correct", [])
            corresponding_exact_equiv_and_correct_gt_triples = find_and_collect_values(exact_equiv_gt_triples_dict, not_match_but_correct)
            not_match_not_exact_equiv_but_correct = [triple for triple in not_match_but_correct if tuple(triple) not in exact_equiv_gt_triples_dict]

            eval_data["corresponding_exact_equiv_and_correct_gt_triples"] = corresponding_exact_equiv_and_correct_gt_triples
            eval_data["not_match_not_exact_equiv_but_correct"] = not_match_not_exact_equiv_but_correct
            
            updated_evaluation_results = [data if data["iteration"] != iteration else eval_data for data in updated_evaluation_results]
            save_evaluation_results(args, evaluator_name, updated_evaluation_results)

        if "evaluator_instance" in locals():
            evaluator_instance.__del__()
            del evaluator_instance
            torch.cuda.empty_cache()
            gc.collect()

def evaluate_conflict_resolution(args):
    """
    为已有 evaluation_results 文件中未执行 Conflict Resolution 的 iteration，补充 Step 11 和 Step 12。
    """
    generated_data = load_generated_data(args)
    generated_data_dict = {data['iteration']: data for data in generated_data}

    for evaluator_name in args.evaluator:
        try:
            evaluation_results = load_evaluation_results(args, evaluator_name=evaluator_name)
        except FileNotFoundError:
            logging.warning(f"Evaluation results for {evaluator_name} not found. Skipping evaluator.")
            continue
        logging.info(f"Loaded evaluation results for evaluator {evaluator_name}.")

        updated_evaluation_results = evaluation_results.copy()

        # conflict_resolution_missing = [
        #     data for data in evaluation_results 
        #     if 'detected_conflict_triple_dict_on_similar' not in data or 
        #        'removal_conflict_triple_dict_on_similar' not in data or
        #        'not_match_not_exact_equiv_but_correct_no_conflict' not in data or
        #        'gt_triples_to_remove' not in data
        # ]

        conflict_resolution_missing = [data for data in evaluation_results]

        if not conflict_resolution_missing:
            logging.info(f"All iterations already contain conflict resolution results for {evaluator_name}.")
            continue

        if evaluator_name in api_models:
            _, _, _, _, _, _, eval_confliction = get_api_function(evaluator_name)
        else:
            EvaluatorClass = get_local_llm_class(evaluator_name)
            evaluator_instance = EvaluatorClass()
            eval_confliction = evaluator_instance.eval_confliction

        for eval_data in conflict_resolution_missing:
            iteration = eval_data["iteration"]
            if iteration < args.start:
                continue  
            logging.info(f"Performing conflict resolution for iteration {iteration} using {evaluator_name}.")

            similar_gt_triples_dict_w_promptlink = eval_data.get("not_match_but_similar_triples_w_promptlink", {})

            if iteration in generated_data_dict:
                chunks = generated_data_dict[iteration]['chunks']
            else:
                logging.warning(f"No corresponding chunks found for iteration {iteration} in generated_data.json. Using empty list.")
                chunks = []

            detected_conflict_triple_dict_on_similar, removal_conflict_triple_dict_on_similar = eval_confliction(similar_gt_triples_dict_w_promptlink, chunks, args.enable)

            removal_conflict_triples = {tuple(item) if not isinstance(item, tuple) else item for v in removal_conflict_triple_dict_on_similar.values() for item in v}            
            not_match_not_exact_equiv_but_correct = eval_data.get('not_match_not_exact_equiv_but_correct', [])
            not_match_not_exact_equiv_but_correct_no_conflict = [triple for triple in not_match_not_exact_equiv_but_correct if tuple(triple) not in removal_conflict_triples]
            gt_triples_to_remove = [triple for triple in removal_conflict_triple_dict_on_similar.keys() if tuple(triple) in removal_conflict_triples]

            eval_data['detected_conflict_triple_dict_on_similar'] = detected_conflict_triple_dict_on_similar
            eval_data['removal_conflict_triple_dict_on_similar'] = removal_conflict_triple_dict_on_similar
            eval_data['not_match_not_exact_equiv_but_correct_no_conflict'] = not_match_not_exact_equiv_but_correct_no_conflict
            eval_data['gt_triples_to_remove'] = gt_triples_to_remove

            updated_evaluation_results = [data if data["iteration"] != iteration else eval_data for data in updated_evaluation_results]
            save_evaluation_results(args, evaluator_name, updated_evaluation_results)

        if "evaluator_instance" in locals():
            evaluator_instance.__del__()
            del evaluator_instance
            torch.cuda.empty_cache()
            gc.collect()
            
def collect_results(args):
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")

    # Load generated_data
    generated_data_path = os.path.join(save_directory, 'generated_data.json')
    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)

    # Initialize results per iteration
    combined_results = []

    # For each iteration in generated_data
    for data in generated_data:
        iteration = data['iteration']
        keywords = data['keywords']

        iteration_result = {
            'iteration': iteration,
            'keywords': keywords,
            'generated_data': data,
            'evaluations': {}
        }

        # Load evaluation results for each evaluator
        for evaluator_name in args.evaluator:
            evaluation_file_path = os.path.join(save_directory, f'{evaluator_name}_evaluation_results.json')
            if os.path.exists(evaluation_file_path):
                with open(evaluation_file_path, 'r', encoding='utf-8') as f:
                    evaluator_evaluations = json.load(f)
                # Find the evaluation result for this iteration
                eval_data = next((item for item in evaluator_evaluations if item['iteration'] == iteration), None)
                if eval_data:
                    iteration_result['evaluations'][evaluator_name] = eval_data
                else:
                    iteration_result['evaluations'][evaluator_name] = None
            else:
                iteration_result['evaluations'][evaluator_name] = None

        combined_results.append(iteration_result)

    # Save combined results to a json file
    combined_results_path = os.path.join(save_directory, 'combined_results.json')
    with open(combined_results_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)
    logging.info(f"Combined results saved to {combined_results_path}")

def calculate_metrics_from_saved_data(args):
    """Calculate metrics from saved generated data and evaluation results."""
    logging.info("Loading generated data and evaluation results for metrics calculation.")
    generated_data, evaluation_results = load_combined_data(args)

    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]
    save_directory = os.path.join(args.save_dir, args.dataset, f"{args.run_id}_{keywords_file_without_extension}_{args.log_file}")
    relevant_triples, relevant_entities, relevant_relations = filter_relevant_kg(args)

    for data in generated_data:
        iteration = data['iteration']
        accumulated_kg_triples = set(triple for d in generated_data[:iteration + 1] for triple in d['kg_triples'])

        for evaluator_name in args.evaluator:
            if evaluator_name not in evaluation_results:
                continue

            recall_precision_calculator = RecallPrecisionCalculator(relevant_triples, relevant_entities, relevant_relations, mode=args.mode)
            metric_logger = MetricLogger(save_path=save_directory, generator_name=args.model, evaluator_name=evaluator_name, save_interval=1)

            # Collect accumulated evaluation data up to this iteration
            accumulated_not_match_but_similar_triples_w_promptlink = []
            accumulated_corresponding_similar_gt_triples_w_promptlink = []
            accumulated_not_match_but_equiv_triples = []
            accumulated_corresponding_equiv_gt_triples = []
            accumulated_not_match_but_exact_equiv_triples = []
            accumulated_corresponding_exact_equiv_gt_triples = []
            accumulated_not_match_but_correct = []
            accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink = []
            accumulated_corresponding_equiv_and_correct_gt_triples = []
            accumulated_corresponding_exact_equiv_and_correct_gt_triples = []

            for eval_data in evaluation_results[evaluator_name]:
                if eval_data['iteration'] <= iteration:
                    accumulated_not_match_but_similar_triples_w_promptlink.extend(eval_data.get('not_match_but_similar_triples_w_promptlink', []))

                    corresponding_similar_gt_triples_w_promptlink = find_and_collect_values(eval_data['not_match_but_similar_triples_w_promptlink'], list(eval_data['not_match_but_similar_triples_w_promptlink']))
                    accumulated_corresponding_similar_gt_triples_w_promptlink.extend(corresponding_similar_gt_triples_w_promptlink)
                    # accumulated_corresponding_similar_gt_triples_w_promptlink.extend(eval_data.get('corresponding_similar_gt_triples_w_promptlink', []))                 # can directly use this if correctly saved in evaluator_evaluation_results.json, but if not, use the above line

                    accumulated_not_match_but_equiv_triples.extend(eval_data.get('not_match_but_equiv_triples', []))
                    accumulated_corresponding_equiv_gt_triples.extend(eval_data.get('corresponding_equiv_gt_triples', []))
                    accumulated_not_match_but_exact_equiv_triples.extend(eval_data.get('not_match_but_exact_equiv_triples', []))
                    accumulated_corresponding_exact_equiv_gt_triples.extend(eval_data.get('corresponding_exact_equiv_gt_triples', []))
                    accumulated_not_match_but_correct.extend(eval_data.get('not_match_but_correct', []))

                    corresponding_similar_and_correct_gt_triples_w_promptlink = find_and_collect_values(eval_data['not_match_but_similar_triples_w_promptlink'], eval_data['not_match_but_correct'])
                    accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink.extend(corresponding_similar_and_correct_gt_triples_w_promptlink)
                    # accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink.extend(eval_data.get('corresponding_similar_and_correct_gt_triples_w_promptlink', []))         # can directly use this if correctly saved in evaluator_evaluation_results.json, but if not, use the above line 

                    accumulated_corresponding_equiv_and_correct_gt_triples.extend(eval_data.get('corresponding_equiv_and_correct_gt_triples', []))
                    accumulated_corresponding_exact_equiv_and_correct_gt_triples.extend(eval_data.get('corresponding_exact_equiv_and_correct_gt_triples', []))

            recall_and_precisions = recall_precision_calculator.calculate_all_recalls_and_precisions(
                accumulated_generated_triples=list(accumulated_kg_triples),
                accumulated_not_match_but_similar_triples_w_promptlink=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_not_match_but_similar_triples_w_promptlink)),
                accumulated_corresponding_similar_gt_triples_w_promptlink=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_corresponding_similar_gt_triples_w_promptlink)),
                accumulated_not_match_but_equiv_triples=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_not_match_but_equiv_triples)),
                accumulated_corresponding_equiv_gt_triples=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_corresponding_equiv_gt_triples)),
                accumulated_not_match_but_exact_equiv_triples=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_not_match_but_exact_equiv_triples)),
                accumulated_corresponding_exact_equiv_gt_triples=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_corresponding_exact_equiv_gt_triples)),
                accumulated_not_match_but_correct=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_not_match_but_correct)),
                accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_corresponding_similar_and_correct_gt_triples_w_promptlink)), 
                accumulated_corresponding_equiv_and_correct_gt_triples=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_corresponding_equiv_and_correct_gt_triples)),
                accumulated_corresponding_exact_equiv_and_correct_gt_triples=list(set(tuple(item) if isinstance(item, list) else item for item in accumulated_corresponding_exact_equiv_and_correct_gt_triples)),
            )
            metric_logger.log_metrics(iteration=iteration, metrics=recall_and_precisions)

    comprehensive_logger = ComprehensiveMetricLogger(save_path=save_directory, generator_name=args.model, evaluators=args.evaluator, save_interval=1)
    comprehensive_logger.plot_comprehensive_metrics()
    logging.info("Metrics plotting completed.")

def plot_all_metrics(args):
    save_root = args.save_dir
    models_evaluators_dict = detect_models_and_evaluators(save_root)

    plot_all_models_metrics(save_root, models_evaluators_dict)
    plot_all_models_triple_recall(save_root, models_evaluators_dict)
    
def save_relevant_kg_to_files(args):
    relevant_triples, relevant_entities, relevant_relations = filter_relevant_kg(args)
    keywords_file_without_extension = os.path.splitext(args.keywords_file)[0]

    triples_file = f'./relevant_triples_{args.dataset}_{keywords_file_without_extension}.txt'
    entities_file = f'./relevant_entities_{args.dataset}_{keywords_file_without_extension}.txt'
    relations_file = f'./relevant_relations_{args.dataset}_{keywords_file_without_extension}.txt'

    with open(triples_file, 'w', encoding='utf-8') as f:
        for triple in relevant_triples:
            f.write(', '.join(triple) + '\n')  # Convert tuple to string with comma-separated values

    with open(entities_file, 'w', encoding='utf-8') as f:
        for entity in relevant_entities:
            f.write(entity + '\n')

    with open(relations_file, 'w', encoding='utf-8') as f:
        for relation in relevant_relations:
            f.write(relation + '\n')

    logging.info(f"Relevant triples saved to {triples_file}")
    logging.info(f"Relevant entities saved to {entities_file}")
    logging.info(f"Relevant relations saved to {relations_file}")

def main():
    args = parse_args()
    set_seed(args.seed)    # Set random seed
    set_logger(args)    # Write logs to checkpoint and console

    # Log all configuration arguments
    logging.info("Configuration arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    if args.mode == 'generate':
        logging.info(f"Starting the knowledge graph generation process for {args.dataset}.")
        generate(args)
    elif args.mode == 'eval':
        logging.info(f"Starting the knowledge graph expansion process for {args.dataset}.")
        evaluate(args)
        collect_results(args)
    elif args.mode == 'generate_and_eval':
        logging.info(f"Starting the knowledge graph expansion process for {args.dataset}.")
        generate(args)
        evaluate(args)
        collect_results(args)
    elif args.mode == 'metric':
        logging.info(f"Calculating metrics for all expanders for generator {args.model}.")
        calculate_metrics_from_saved_data(args)
    elif args.mode == 'mask':
        logging.info(f"Getting relevant masked KG for {args.dataset}.")
        save_relevant_kg_to_files(args)
    elif args.mode == "eval_exact_equiv":
        logging.info(f"Starting `exact_equiv` evaluation process for {args.dataset}.")
        evaluate_exact_equiv(args)
    elif args.mode == 'plot_all':
        logging.info(f"Calculating metrics for all models in {args.save_dir}.")
        plot_all_metrics(args)
    elif args.mode == 'conflict_resolve':
        logging.info(f"Resolving conflicts for {args.dataset}.")
        evaluate_conflict_resolution(args)


if __name__ == "__main__":
    main()