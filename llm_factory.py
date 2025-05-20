"""
model list:
'mistral-7b', 'llama-3-8b', 'llama-3-70b', 'llama-3.1-405b', 'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'gemini'
"""

def get_local_llm_class(model):
    if model == 'phi-3.5-mini':
        from phiGen import LocalLLM
    elif model == 'mistral-7b':
        from mistralGen import LocalLLM
    elif model == 'llama-3-8b':
        from llama8bGen import LocalLLM
    elif model == 'llama-3-70b':
        from llama70bGen import LocalLLM
    elif model == 'llama-3.1-405b':
        from llama405bGen import LocalLLM
    else:
        raise ValueError("Invalid model type")
    return LocalLLM

def get_api_function(model):
    if model == 'gpt-3.5-turbo':
        from gptturboGen import generate_kg, get_similar_triples_dict_w_promptlink, eval_similar_w_promptlink, eval_accuracy, eval_equivalence, eval_exact_equiv, eval_confliction  
    elif model == 'gpt-4o-mini':
        from gpt4ominiGen import generate_kg, get_similar_triples_dict_w_promptlink, eval_similar_w_promptlink, eval_accuracy, eval_equivalence, eval_exact_equiv, eval_confliction 
    elif model == 'gpt-4o':
        from gpt4oGen import generate_kg, get_similar_triples_dict_w_promptlink, eval_similar_w_promptlink, eval_accuracy, eval_equivalence, eval_exact_equiv, eval_confliction  
    elif model == 'gemini':
        from geminiGen import generate_kg, get_similar_triples_dict_w_promptlink, eval_similar_w_promptlink, eval_accuracy, eval_equivalence, eval_exact_equiv, eval_confliction 
    else:
        raise ValueError("Invalid generator/evaluator model type")
    return generate_kg, get_similar_triples_dict_w_promptlink, eval_similar_w_promptlink, eval_accuracy, eval_equivalence, eval_exact_equiv, eval_confliction  

