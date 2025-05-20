import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from jinjaRead import jinjaReadCriteria
from logging import getLogger
from typing import List, Tuple, Dict
import pdb
from llm_utils import clean_symbols, read_triples_from_file, parse_triples_from_file, parse_triples_from_llm, clean_entity, clean_relation, eval_match, parse_keywords_from_llm, parse_conflict_and_removal 
from enableIns import graphusionCriteria, graphjudgerCriteria, graphjudger_keyword_extraction, graphjudger_chunks_denoising, llmkg_conflict_resolution_prompt, graphusion_conflict_resolution_prompt, moreCriteria
from tqdm import tqdm


logger = getLogger(__name__)
# pdb.set_trace()


hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables.")

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
local_model_path = "hfmodels/Meta-Llama-3-70B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running on {device}...")
print(f"Running on {device}...")

# change the quantization type if needed
quantize_type = '4bit'

# specify how to quantize the model
nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
)

my8_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.float16,
)

half_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=torch.float16,
) 

# set the quantization configuration
quantization_config = None

if quantize_type == '4bit':
    quantization_config = nf4_config
elif quantize_type == '8bit':
    quantization_config = my8_config
elif quantize_type == 'half':
    quantization_config = half_config


class LocalLLM:
    def __init__(self):
        if quantization_config:
            logger.info(f"Model is quantified using configuration: {quantization_config}")
            # Shrinking down Mixtral using quantization
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, quantization_config=quantization_config, device_map="auto", force_download=True)
            # quantization + Flash Attention
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, quantization_config=quantization_config, device_map="auto", force_download=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, token=hf_token, quantization_config=quantization_config, device_map="auto", force_download=False, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        else:
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, device_map="auto", force_download=True)
            # Speeding up Mixtral by using Flash Attention
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, token=hf_token, device_map="auto", force_download=False, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

        # self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, token=hf_token)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def eval_accuracy(self, not_matched_triples: List[Tuple[str, str, str]], chunks: List[str], enable=['none']) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Evaluate the accuracy of each not matched triple individually using the LLM response.
        Each triple is evaluated separately to improve accuracy.
        """
        not_match_but_correct = []
        not_match_and_wrong = []
        keywords = ['true', 'yes', 'correct', 'right', 'equivalent']    # Keywords to look for in the response to evaluate equivalence

        for gen_triple in not_matched_triples:
            if 'correctness_got' in enable:
                add = "Please consider in the following way: Step 1: Identify the key entities and relationships in the document segment. Step 2: Extract the context around the identified entities. Step 3: Infer if the relationship is directly supported by the text. Step 4: Verify the accuracy of the proposed knowledge graph triple."
            else:
                add = ""

            messages = [
                {"role": "system", "content": f"You are an expert at evaluating the accuracy of scientific knowledge graph based on the provided documents segment.{add}"},
                {"role": "user", "content": f"Based on the document segments provided, please determine the correctness of the knowledge graph triple. The knowledge graph triple:\n{gen_triple}\nThe document segment:\n{chunks}\nRespond with only one word: 'TRUE' or 'FALSE' and do not provide any explanation or reasoning."},
            ]

            input_ids = self.tokenizer.apply_chat_template(messages, max_length=4096, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)
            generated_ids = self.model.generate(input_ids, max_new_tokens=30, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)
            response = generated_ids[0][input_ids.shape[-1]:]
            raw_answer = self.tokenizer.decode(response, skip_special_tokens=True)

            if any(keyword in raw_answer.lower() for keyword in keywords):
                not_match_but_correct.append(gen_triple)
            else:
                not_match_and_wrong.append(gen_triple)

        print(f'\n--- Eval accuracy --- \nIn {len(not_matched_triples)} not matched triples, model judges:')
        print(f'{len(not_match_but_correct)} not match but correct triples:\n{not_match_but_correct}')
        print(f'{len(not_match_and_wrong)} not match and wrong triples:\n{not_match_and_wrong}')

        return not_match_but_correct, not_match_and_wrong

    def eval_confliction(self, similar_gt_triples_dict, chunks: List[str], enable=['none']) -> Tuple[dict, dict]:
        """
        If a head-tail pair of triples has multiple relations, ask the model to evaluate the conflict.
        Only activated when 'conflict_resolve' is in enable.
        
        Returns:
            - detected_conflict_triple_dict: Dict[ground_truth_triple, List[conflicting_triples]]
            - removal_conflict_triple_dict: Dict[ground_truth_triple, List[triples_to_remove]]
        """
        if 'conflict_resolve' in enable:
            conflict_resolution_prompt = llmkg_conflict_resolution_prompt
        elif 'conflict_graphusion' in enable:
            conflict_resolution_prompt = graphusion_conflict_resolution_prompt
        else:
            return {}, {}

        detected_conflict_triple_dict = {}
        removal_conflict_triple_dict = {}

        for gt_triple, gen_triples in similar_gt_triples_dict.items():
            all_triples = [gt_triple] + gen_triples
            
            # substep 1: Ask LLM to list conflicting triples
            triples_str = "\n".join([f"({h}, {r}, {t})" for (h, r, t) in all_triples])
            chunks_str = "\n".join(chunks)
            messages = [{"role": "user", "content": conflict_resolution_prompt.format(triples_str=triples_str, chunks_str=chunks_str)},]

            input_ids = self.tokenizer.apply_chat_template(messages, max_length=4096, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)
            generated_ids = self.model.generate(input_ids, max_new_tokens=500, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)
            response = generated_ids[0][input_ids.shape[-1]:]
            raw_conflict_output = self.tokenizer.decode(response, skip_special_tokens=True)

            # substep 2: Parse the output to get the conflicting triples 
            conflict_triples, removal_triples = parse_conflict_and_removal(raw_conflict_output)
            if conflict_triples:
                detected_conflict_triple_dict[gt_triple] = conflict_triples
            if removal_triples:
                removal_conflict_triple_dict[gt_triple] = removal_triples

        print(f'\n--- Eval confliction ---\nProcessed {len(similar_gt_triples_dict)} groups.')
        print(f'Found {sum(len(v) for v in detected_conflict_triple_dict.values())} conflicting triples across all groups.')
        print(f'Proposed to remove {sum(len(v) for v in removal_conflict_triple_dict.values())} triples to resolve conflicts.')

        return detected_conflict_triple_dict, removal_conflict_triple_dict

    def eval_exact_equiv(self, equivalent_gt_triples_dict) -> Dict[Tuple[str, str, str], List[Tuple[str, str, str]]]:
        """
        Ask llm to evaluate each equivalent triples and return the ones that are exactly equivalent. Note that if there is no exact equivalence found, the triple is skipped. 
        """
        exact_equiv_gt_triples_dict = {}
        keywords = ['true', 'yes', 'correct', 'right', 'equivalent']    # Keywords to look for in the response to evaluate equivalence

        for gen_triple, equiv_gt_triples in equivalent_gt_triples_dict.items():
            exact_equiv_gt_triples_dict[gen_triple] = []
            for equiv_gt_triple in equiv_gt_triples:
                messages = [
                    {"role": "system", "content": (
                            "You are a medical knowledge expert. Your task is to determine whether two given triples are exactly equivalent in meaning, "
                            "based on the following **strict criteria**: \n"
                            "1. **Entity Equivalence**: The subject and object in the triples must refer to the same medical entities or concepts, "
                            "either directly or through synonyms (e.g., 'blood sugar' and 'glucose levels'). Use standard medical knowledge.\n"
                            "2. **Predicate Equivalence**: The relationship (predicate) must describe the exact same logical or semantic connection between the entities. "
                            "Synonyms are acceptable if they represent identical relationships (e.g., 'regulates' and 'modulates' in the context of insulin).\n"
                            "3. **Context Consistency**: Both triples must convey the same fact or mechanism in the medical domain. Small wording differences are acceptable, "
                            "but any differences in meaning or implication must lead to a 'FALSE' judgment.\n\n"
                            "If the triples meet all three criteria, answer 'TRUE'. Otherwise, answer 'FALSE'.\n\n"
                            "Here are examples to guide you:\n"
                            "- TRUE: <insulin, regulates, blood sugar> and <insulin, modulates, glucose levels>\n"
                            "- FALSE: <insulin, regulates, blood sugar> and <insulin, influences, blood sugar> (relationship is not identical)\n"
                            "- FALSE: <insulin, regulates, blood sugar> and <insulin, regulates, cholesterol> (entities differ)\n"
                        )
                    },
                    {"role": "user", "content": f"Please judge if the following two triples are exactly equivalent. Triple 1: {gen_triple}, Triple 2: {equiv_gt_triple}. Please answer by providing 'TRUE' or 'FALSE' value."},
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, max_length=300, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)
                generated_ids = self.model.generate(input_ids, max_new_tokens=300, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)
                raw_answer = generated_ids[0][input_ids.shape[-1]:]
                raw_answer = self.tokenizer.decode(raw_answer, skip_special_tokens=True)

                if any(keyword in raw_answer.lower() for keyword in keywords):
                    exact_equiv_gt_triples_dict[gen_triple].append(equiv_gt_triple)
                #     logger.info(f"[STEP 8] {gen_triple} is similar to {equiv_gt_triple}")
                # else:
                #     logger.info(f"[STEP 8] {gen_triple} is not similar to {equiv_gt_triple}")
        exact_equiv_gt_triples_dict = {k: v for k, v in exact_equiv_gt_triples_dict.items() if v}

        return exact_equiv_gt_triples_dict
        
    def eval_equivalence(self, similar_gt_triples_dict_w_promptlink) -> Dict[Tuple[str, str, str], List[Tuple[str, str, str]]]:
        """
        Ask llm to evaluate each similar triples and return the ones that are equivalent. Note that if there is no equivalence found, the triple is skipped. 
        """
        equivalent_gt_triples_dict = {}
        keywords = ['true', 'yes', 'correct', 'right', 'equivalent']    # Keywords to look for in the response to evaluate equivalence

        for gen_triple, similar_gt_triples in similar_gt_triples_dict_w_promptlink.items():
            equivalent_gt_triples_dict[gen_triple] = []
            for similar_gt_triple in similar_gt_triples:
                messages = [
                    {"role": "system", "content": "You are an expert at evaluating scientific knowledge graphs equivalence who always responds by providing 'TRUE' or 'FALSE' value."},
                    {"role": "user", "content": f"Please judge if the following two triples are equivalent. Triple 1: {gen_triple}, Triple 2: {similar_gt_triple}."},
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, max_length=300, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)
                generated_ids = self.model.generate(input_ids, max_new_tokens=300, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)
                raw_answer = generated_ids[0][input_ids.shape[-1]:]
                raw_answer = self.tokenizer.decode(raw_answer, skip_special_tokens=True)

                if any(keyword in raw_answer.lower() for keyword in keywords):
                    equivalent_gt_triples_dict[gen_triple].append(similar_gt_triple)
                #     logger.info(f"[STEP 7] {gen_triple} is similar to {similar_gt_triple}")
                # else:
                #     logger.info(f"[STEP 7] {gen_triple} is not similar to {similar_gt_triple}")
        equivalent_gt_triples_dict = {k: v for k, v in equivalent_gt_triples_dict.items() if v}

        return equivalent_gt_triples_dict
    
    def get_similar_entities_dict_w_promptlink(self, keywords_expander: str, k_similar_entities_dict_w_sapbert: Dict[str, List[str]], dataset: str) -> List[str]:
        """
        Input: k_similar_entities_dict_w_sapbert
        Output: similar_entities_dict_w_promptlink
        """
        _, filter_entity_criteria, _, _ = jinjaReadCriteria(keywords_expander, dataset)
        similar_entities_dict_w_promptlink = {}
        keywords = ['true', 'yes', 'correct', 'right', 'equivalent']    # Keywords to look for in the response to evaluate equivalence

        for gen_entity, similar_gt_entities in k_similar_entities_dict_w_sapbert.items():
            similar_entities_dict_w_promptlink[gen_entity] = []
            for similar_gt_entity in similar_gt_entities:
                messages = [
                    {"role": "system", "content": f"You are an expert at evaluating entity similarity in scientific knowledge graphs based on these criteria:\n{filter_entity_criteria}. You always respond by providing 'TRUE' or 'FALSE' value."},
                    {"role": "user", "content": f"Please judge if the following two entities are similar. The query entity: {gen_entity}, the candidate entity: {similar_gt_entity}."},
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, max_length=300, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)
                generated_ids = self.model.generate(input_ids, max_new_tokens=30, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)
                raw_answer = generated_ids[0][input_ids.shape[-1]:]
                raw_answer = self.tokenizer.decode(raw_answer, skip_special_tokens=True)

                if any(keyword in raw_answer.lower() for keyword in keywords):
                    similar_entities_dict_w_promptlink[gen_entity].append(similar_gt_entity)

        similar_entities_dict_w_promptlink = {k: v for k, v in similar_entities_dict_w_promptlink.items() if v}
        return similar_entities_dict_w_promptlink

    def get_similar_triples_dict_w_promptlink(self, not_matched_triples, k_similar_entities_dict_w_sapbert, dataset: str, strictness: str = 'strict', keywords_expander: str = 'none') -> Dict[str, List[str]]:
        """
        Find the similar entities for each query entity in not_matched_triples using PromptLink. 
        Then find related gt triples only considering the entity similarity. Note that this function only consider entity similarity, not relation similarity: 
        strict: both head and tail are similar; 
        loose: either head or tail is similar;
        """
        similar_entities_dict = self.get_similar_entities_dict_w_promptlink(keywords_expander, k_similar_entities_dict_w_sapbert, dataset)

        # Filter 1, reduce range: get all similar gt entities for query entities in not_matched_triples
        all_similar_gt_entities = set()
        for entities in similar_entities_dict.values():
            all_similar_gt_entities.update(entities)

        # Filter 2, find related gt triples
        gt_triples = read_triples_from_file(dataset, file_name='triples.txt')    # This might need large memory if the KG is large
        similar_gt_triples = []
        for triple in gt_triples:
            head, relation, tail = [part.strip() for part in triple.replace('<', '').replace(">", '').split(', ')]
            if strictness == 'lenient':
                if head in all_similar_gt_entities or tail in all_similar_gt_entities:
                    similar_gt_triples.append((head, relation, tail))
            elif strictness == 'strict':
                if head in all_similar_gt_entities and tail in all_similar_gt_entities:
                    similar_gt_triples.append((head, relation, tail))
            else:
                raise ValueError(f"Invalid strictness value: {strictness}")

        # Filter 3, find related gt triples for each generated triple
        similar_gt_triples_dict = {}
        for gen_triple in not_matched_triples:
            head, rel, tail = gen_triple
            similar_gt_entities = set()
            if head in similar_entities_dict:
                similar_gt_entities.update(similar_entities_dict[head])
            if tail in similar_entities_dict:
                similar_gt_entities.update(similar_entities_dict[tail])

            similar_gt_triples_dict[gen_triple] = []
            for gt_triple in similar_gt_triples:
                gt_head, gt_rel, gt_tail = gt_triple
                if strictness == 'lenient':
                    if gt_head in similar_gt_entities or gt_tail in similar_gt_entities:
                        similar_gt_triples_dict[gen_triple].append(gt_triple)
                elif strictness == 'strict':
                    if gt_head in similar_gt_entities and gt_tail in similar_gt_entities:
                        similar_gt_triples_dict[gen_triple].append(gt_triple)

        similar_gt_triples_dict = {k: v for k, v in similar_gt_triples_dict.items() if v}
        return similar_entities_dict, similar_gt_triples_dict

    def eval_similar_w_promptlink(self, keywords_expander, similar_gt_triples_dict, dataset: str) -> Dict[Tuple[str, str, str], List[Tuple[str, str, str]]]:
        """
        Ask llm to confirm each similar triples and return the ones that are correct. Note that this function consider the whole triple similariry. 
        """
        _, filter_entity_criteria, _, _ = jinjaReadCriteria(keywords_expander, dataset)
        similar_gt_triples_dict_w_promptlink = {}
        keywords = ['true', 'yes', 'correct', 'right', 'equivalent']    # Keywords to look for in the response to evaluate equivalence

        for gen_triple, similar_gt_triples in similar_gt_triples_dict.items():
            similar_gt_triples_dict_w_promptlink[gen_triple] = []
            for similar_gt_triple in similar_gt_triples:
                messages = [
                    {"role": "system", "content": f"You are an expert at evaluating triples similarity in scientific knowledge graphs based on these criteria:\n{filter_entity_criteria}. You always respond by providing 'TRUE' or 'FALSE' value."},
                    {"role": "user", "content": f"Please judge if the following two triples are similar. The query triple: {gen_triple}, the candidate triple: {similar_gt_triple}"},
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, max_length=300, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)
                generated_ids = self.model.generate(input_ids, max_new_tokens=30, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)
                raw_answer = generated_ids[0][input_ids.shape[-1]:]
                raw_answer = self.tokenizer.decode(raw_answer, skip_special_tokens=True)

                if any(keyword in raw_answer.lower() for keyword in keywords):
                    similar_gt_triples_dict_w_promptlink[gen_triple].append(similar_gt_triple)
                #     logger.info(f"[STEP 7] {gen_triple} is similar to {similar_gt_triple}")
                # else:
                #     logger.info(f"[STEP 7] {gen_triple} is not similar to {similar_gt_triple}")
        similar_gt_triples_dict_w_promptlink = {k: v for k, v in similar_gt_triples_dict_w_promptlink.items() if v}
        return similar_gt_triples_dict_w_promptlink

    def iterative_denoising_and_keyword_extraction(self, keywords: List[str], chunks: List[str], iteration_times: int = 3) -> Tuple[List[str], List[List[str]]]:
        """
        Iteratively denoise document segments and extract keywords.
        """
        current_keywords = keywords
        current_chunks = chunks

        for iteration in range(1, iteration_times + 1):
            logger.info(f"[Iteration {iteration}] Starting keywords extraction.")

            # Step 1: Extract keywords for each chunk
            keyword_extraction = graphjudger_keyword_extraction
            for chunk in tqdm(current_chunks, desc=f"Keywords Extraction Iter {iteration}"):
                messages = [
                    {"role": "system", "content": f"You are a scientific knowledge graph constructor who is trained to transform the text into a list of important keywords based on these criteria:\n{keyword_extraction}."},
                    {"role": "user", "content": f"Please transform the text into a list of important keywords.\nText:\n{chunk}."},
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, max_length=4096, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)    # Encoding the input messages
                generated_ids = self.model.generate(input_ids, max_new_tokens=2048, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)   # Generating the output
                response = generated_ids[0][input_ids.shape[-1]:]
                out = self.tokenizer.decode(response, skip_special_tokens=True)
                extracted_keywords = parse_keywords_from_llm(out)
                current_keywords.append(extracted_keywords)

            # Step 2: Use extracted keywords to denoise the chunks
            chunks_denoising = graphjudger_chunks_denoising
            for chunk, keywords in tqdm(zip(current_chunks, current_keywords), desc=f"Denoising Iter {iteration}", total=len(current_chunks)):
                denoised_chunks = []
                messages = [
                    {"role": "system", "content": f"You are a scientific knowledge graph constructor who is trained to denoise the raw text with the given keywords based on these criteria:\n{chunks_denoising}."},
                    {"role": "user", "content": f"Please denoise the raw text with the given keywords.\nText:\n{chunk}\nKeywords:\n{keywords}."},
                ]
                input_ids = self.tokenizer.apply_chat_template(messages, max_length=4096, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)    # Encoding the input messages
                generated_ids = self.model.generate(input_ids, max_new_tokens=2048, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)   # Generating the output
                response = generated_ids[0][input_ids.shape[-1]:]
                out = self.tokenizer.decode(response, skip_special_tokens=True)
                denoised_chunks.append(out)

            # After denoising, set current chunks to denoised results for next iteration
            current_chunks = denoised_chunks

        return current_keywords, current_chunks

    def generate_kg(self, keywords_expander, dataset: str, keywords: List[str], chunks: List[str], enable=['none']) -> str:
        """
        Generate the knowledge graph based on the provided keywords and document segments.
        """
        if 'criteria_graphusion' in enable:
            generate_kg_criteria = graphusionCriteria     # use this for Graphusion baseline 
        elif 'criteria_graphjudger' in enable:
            generate_kg_criteria = graphjudgerCriteria     # use this for graphjudger baseline 
        else:
            generate_kg_criteria, _, _, _ = jinjaReadCriteria(keywords_expander, dataset)

        more = moreCriteria.strip() if 'criteria_more' in enable else ''

        if 'denoising_doc' in enable:
            keywords, chunks = self.iterative_denoising_and_keyword_extraction(keywords, chunks)

        messages = [
            {"role": "system", "content": f"You are a scientific knowledge graph constructor who is trained to generate knowledge graphs based on these criteria: {generate_kg_criteria}\n{more}."},
            {"role": "user", "content": f"Please generate knowledge graph triples based on the provided keywords and document segments. Keywords: {keywords}. Document segments: {chunks}."},
        ]

        input_ids = self.tokenizer.apply_chat_template(messages, max_length=4096, add_generation_prompt=True, return_tensors="pt", padding=True).to(device)    # Encoding the input messages
        # print(input_ids)
        input_token_count = input_ids.shape[1] # Counting the input tokens
        logger.info(f"[STEP 4] KG generation input token count: {input_token_count}")

        generated_ids = self.model.generate(input_ids, max_new_tokens=2048, eos_token_id=self.terminators, use_cache=True, do_sample=True, temperature=0.6, top_p=0.9,)   # Generating the output
        # print(generated_ids)
        output_token_count = generated_ids.shape[1]  # Counting the output tokens
        logger.info(f"[STEP 4] KG generation output token count: {output_token_count}")

        response = generated_ids[0][input_ids.shape[-1]:]
        out = self.tokenizer.decode(response, skip_special_tokens=True)

        # print("\nModel's response:\n", out)
        return out

    def __del__(self):
        try:
            if hasattr(self, 'model') and self.model is not None:
                self.model.cpu()  
                del self.model  
            if hasattr(self, 'tokenizer'):
                del self.tokenizer  
            if hasattr(self, 'terminators'):
                del self.terminators  
            torch.cuda.empty_cache()  
        except Exception as e:
            logger.warning(f"Error during cleanup in __del__: {e}")


if __name__ == "__main__":
    # kg_triples = generate_kg(dataset='base', keywords='herb', chunks='Herbs are natural medicine that can cure human.')
    # print(f"[STEP 4] Generated KG triples:\n{kg_triples}")
    # print(f"[STEP 4] Generated KG triples include {len(kg_triples)} tokens.")

    # deduplicated_and_pruned_triples = deduplicate_and_prune_kg(dataset='base', accumulated_triples='herb    belongs_to  TCM')
    # print(f'[STEP 5] Deduplicated and pruned KG triples:\n{deduplicated_and_pruned_triples}')
    # print(f"[STEP 5] Deduplicated and pruned accumulated triples include {len(deduplicated_and_pruned_triples)} tokens.")

    # score_with_reason = evaluate_kg(dataset='base', deduplicated_and_pruned_triples='herb    belongs_to  TCM')
    # print(f"[STEP 7] Score with reason:\n{score_with_reason}")
    
    keywords = ['herb']

    chunks = '''
    I know I love you.
    '''

    kg_string = '''
    (I, love, you) +
    (Ginseng, 'has efficacy', boosts immunity) +
    (Coptis, is used for, 'enteritis') +
    ('Salvia miltiorrhiza'，'belongs to'，'blood-tonifying medicinal category') +

    ('I','love','you') +
    ('Ginseng', 'has_efficacy', 'boosts immunity') +
    ('Coptis', 'is_used_for', 'enteritis') +
    ('Salvia miltiorrhiza', 'belongs_to', 'blood-tonifying medicinal category') +

    ('I', 'like', 'you') + 

    1. (head1, has_known_target_protein_of, cyclic AMP-dependent protein kinase) +
    Some text "'head2', 'relation', 'tail'" +
    (head3 , relation , tail) +
    [head4，relation，tail] +

    and some more "(head5, 'relation', 'tail')" +
    "<'head6', 'relation', 'tail'>", +
    "['head7' 'relation' 'tail']"
    ('head8','relation', '2-azaniumylethyl [(2R)-2,3-di(dodecanoyloxy)propyl] phosphate(90657571)') +
    (\head9 , relation , tail)
    '''

    # kg_string = generate_kg('base', keywords, chunks)

    # kg_triples = read_triples_from_file('base')

    kg_triples = parse_triples_from_llm(kg_string)
    matched_triples, not_matched_triples = eval_match(kg_triples, 'base')
    # not_match_but_correct, not_match_and_wrong = eval_accuracy(not_matched_triples, chunks)

    # print(clean_relation('the_hepatitis_B_virus-infection'))

    # parse_triples_from_file('base')


    # Example usage:
    not_matched_triples = [
        'N4 categories, decreases, AST',
        'N4 categories, decreases, ALT',
        'N4 categories, increases, total_protein',
        'N4 categories, increases, serum_albumin',
        'N4 categories, increases, globulin',
        'Random_forest_method, analyzes, environmental_variables'
    ]
    similar_triples_w_promptlink, _ = eval_similar_w_promptlink(keywords_expander='none', not_matched_triples=not_matched_triples, dataset='UMLS')