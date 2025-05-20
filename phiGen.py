import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import os
from jinjaRead import jinjaReadCriteria
from llm_utils import clean_symbols, read_triples_from_file, parse_triples_from_file, parse_triples_from_llm, clean_entity, clean_relation, eval_match
from logging import getLogger
from typing import List, Tuple, Dict
import pdb


logger = getLogger(__name__)
# pdb.set_trace()


hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables.")

model_id = "microsoft/Phi-3.5-mini-instruct"
local_model_path = "hfmodels/Phi-3.5-mini-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running on {device}...")
print(f"Running on {device}...")

# change the quantization type if needed
quantize_type = 'none'

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
            print(f"Model is quantified using configuration: {quantization_config}")
            # Shrinking down Mixtral using quantization
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, quantization_config=quantization_config, device_map="auto")
            # quantization + Flash Attention
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, token=hf_token, quantization_config=quantization_config, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2")
        else:
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, device_map="auto")
            # Speeding up Mixtral by using Flash Attention
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, token=hf_token, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2")

        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, token=hf_token)

        # 适用于没有专门填充令牌的自回归模型
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # 使用eos_token_id作为pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def ask_phi(self, messages):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer) 
        generation_args = {"max_new_tokens": 2048, "return_full_text": False, "temperature": 0.0, "do_sample": False,} 
        output = pipe(messages, **generation_args) 
        output = output[0]['generated_text'] 
        return output.strip()

    def remove_prompt(self, output_text: str) -> str:
        """
        Extracts the content after the last occurrence of "[/INST]" from the model's output.
        
        :param output_text: The complete text output from the model.
        :return: The substring after the last "[/INST]".
        """
        # find the last occurrence of "[/INST]"
        last_inst_index = output_text.rfind("[/INST]")

        # if found, return the substring after it; otherwise, return the entire text
        if last_inst_index != -1:
            # add len("[/INST]") to start from the position after the last "[/INST]"
            return output_text[last_inst_index + len("[/INST]"):].strip()
        else:
            # if the last "[/INST]" was not found, it may be an error, return the original text or you can customize error handling
            return output_text

    def eval_accuracy(self, not_matched_triples: List[Tuple[str, str, str]], chunks: List[str], enable=['none']) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Evaluate the accuracy of each not matched triple individually using the LLM response.
        Each triple is evaluated separately to improve accuracy.
        """
        not_match_but_correct = []
        not_match_and_wrong = []
        keywords = ['true', 'yes', 'correct', 'right', 'equivalent']    # Keywords to look for in the response to evaluate equivalence

        for gen_triple in not_matched_triples:
            messages = [ 
                {"role": "system", "content": "You are an expert."}, 
                {"role": "user", "content": "Please judge if the following knowledge graph triple is correct or not based on the provided document segment. Please answer by providing 'TRUE' or 'FALSE' value."}, 
                {"role": "assistant", "content": "Okay, which knowledge graph triple and document segment is it?"}, 
                {"role": "user", "content": f"The knowledge graph triple:\n{gen_triple}\nThe document segment:\n{chunks}."}, 
            ] 
            raw_answer = self.ask_phi(messages)
            # raw_answer = self.remove_prompt(raw_answer)

            if any(keyword in raw_answer.lower() for keyword in keywords):
                not_match_but_correct.append(gen_triple)
            else:
                not_match_and_wrong.append(gen_triple)

        print(f'\n--- Eval accuracy --- \nIn {len(not_matched_triples)} not matched triples, model judges:')
        print(f'{len(not_match_but_correct)} not match but correct triples:\n{not_match_but_correct}')
        print(f'{len(not_match_and_wrong)} not match and wrong triples:\n{not_match_and_wrong}')

        return not_match_but_correct, not_match_and_wrong

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
                    )},
                    {"role": "user", "content": "Please judge if the following two triples are equivalent. Please answer by providing 'TRUE' or 'FALSE' value."},
                    {"role": "assistant", "content": "Understood. Please provide the triples you want me to evaluate."},
                    {"role": "user", "content": f"Triple 1: {gen_triple}, Triple 2: {equiv_gt_triple}. Please answer by providing 'TRUE' or 'FALSE' value."},
                ]
                raw_answer = self.ask_phi(messages)
                print(f"raw_answer: {raw_answer}")

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
                    {"role": "system", "content": "You are an expert."}, 
                    {"role": "user", "content": "Please judge if the following two triples are equivalent. Please answer by providing 'TRUE' or 'FALSE' value."}, 
                    {"role": "assistant", "content": "Okay, which two triples do you want me to compare?"}, 
                    {"role": "user", "content": f"Triple 1: {gen_triple}, Triple 2: {similar_gt_triple}."}, 
                ] 
                raw_answer = self.ask_phi(messages)
                # raw_answer = self.remove_prompt(raw_answer)
                print(f"raw_answer: {raw_answer}")

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
                    {"role": "system", "content": "You are an expert."}, 
                    {"role": "user", "content": "Please judge if the following query entity and the candidate entity are similar and ensure that your decision adheres to these criteria:\n{filter_entity_criteria}. Please answer by providing 'TRUE' or 'FALSE' value."}, 
                    {"role": "assistant", "content": "Okay, which entity and candidate entity do you want me to compare?"}, 
                    {"role": "user", "content": f"The query entity: {gen_entity}, the candidate entity: {similar_gt_entity}."}, 
                ] 
                raw_answer = self.ask_phi(messages)
                # raw_answer = self.remove_prompt(raw_answer)
                print(f"raw_answer: {raw_answer}")

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
                    {"role": "system", "content": "You are an expert."}, 
                    {"role": "user", "content": "Please judge if the following two triples are similar and ensure that your decision adheres to these criteria:\n{filter_entity_criteria}. Please answer by providing 'TRUE' or 'FALSE' value."}, 
                    {"role": "assistant", "content": "Okay, which two triples do you want me to compare?"}, 
                    {"role": "user", "content": f"The query triple: {gen_triple}, the candidate triple: {similar_gt_triple}."}, 
                ] 
                raw_answer = self.ask_phi(messages)
                # raw_answer = self.remove_prompt(raw_answer)
                print(f"raw_answer: {raw_answer}")

                if any(keyword in raw_answer.lower() for keyword in keywords):
                    similar_gt_triples_dict_w_promptlink[gen_triple].append(similar_gt_triple)
                #     logger.info(f"[STEP 7] {gen_triple} is similar to {similar_gt_triple}")
                # else:
                #     logger.info(f"[STEP 7] {gen_triple} is not similar to {similar_gt_triple}")
        similar_gt_triples_dict_w_promptlink = {k: v for k, v in similar_gt_triples_dict_w_promptlink.items() if v}
        return similar_gt_triples_dict_w_promptlink

    def generate_kg(self, keywords_expander: str, dataset: str, keywords: List[str], chunks: List[str]) -> str:
        """
        Generate the knowledge graph based on the provided keywords and document segments.
        """
        generate_kg_criteria, _, _, _ = jinjaReadCriteria(keywords_expander, dataset)
        messages = [ 
            {"role": "system", "content": "You are a scientific researcher."}, 
            {"role": "user", "content": f"Please generate a knowledge graph starting from the provided keywords and document segments. Keywords: {keywords}. Document segments: {chunks}."}, 
            {"role": "assistant", "content": "To better assist you, could you specify your criteria for generating the knowledge graph?"}, 
            {"role": "user", "content": f"Ensure the knowledge graph adheres to these criteria: {generate_kg_criteria}."}, 
            {"role": "assistant", "content": "Understood. Do you need my answer to include any specific formats or visual representations?"},
            {"role": "user", "content": "No further requirements. Just generate the knowledge graph, focusing solely on the information integration without additional explanations or textual outputs."}
        ] 
        out = self.ask_phi(messages)
        # out = self.remove_prompt(out)
        print("\nModel's response:===============================\n", out)
        return out

    def __del__(self):
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            # 记录日志以便调试，但不会影响程序继续运行
            logger.warning(f"Error during cleanup in __del__: {e}")


if __name__ == "__main__":
    
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