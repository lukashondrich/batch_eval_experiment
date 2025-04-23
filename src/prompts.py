"""
Prompt templates for batch evaluation experiment.
"""

def get_dimension_descriptions():
    """Return consistent dimension descriptions for all prompts."""
    return """
Evaluate each language tutoring message on these dimensions (0.0 = low, 1.0 = high):

1. Lexical Complexity (LC): 
   - 0.0: Elementary vocabulary, common words
   - 1.0: Advanced, specialized terminology

2. Construction Complexity (CC): 
   - 0.0: Simple sentences, basic structures
   - 1.0: Complex embedded clauses, advanced constructions

3. Formality Level (FL): 
   - 0.0: Very informal, conversational
   - 1.0: Highly formal, academic

4. Socratic Approach (SA): 
   - 0.0: Direct statement, telling information
   - 1.0: Question-based exploration, guiding discovery
"""

def get_baseline_batch_prompt(messages):
    """
    Standard batch evaluation prompt.
    All messages evaluated at once.
    """
    dimension_descriptions = get_dimension_descriptions()
    
    messages_text = "\n\n".join([f"Message {i+1}: \"{msg}\"" for i, msg in enumerate(messages)])
    
    prompt = f"""You are an expert language teacher evaluating tutoring messages.
{dimension_descriptions}

Here are the messages to evaluate:

{messages_text}

For each message, provide scores as a JSON array with exactly 4 numbers between 0.0 and 1.0:
[LC_score, CC_score, FL_score, SA_score]

Your response should be a valid JSON object with message numbers as keys:
{{
  "1": [LC_score, CC_score, FL_score, SA_score],
  "2": [LC_score, CC_score, FL_score, SA_score],
  ...
}}
"""
    return prompt


def get_independent_prompt(message, message_id):
    """
    Independent evaluation prompt.
    One message at a time.
    """
    dimension_descriptions = get_dimension_descriptions()
    
    prompt = f"""You are an expert language teacher evaluating a tutoring message.
{dimension_descriptions}

Message: "{message}"

Provide scores as a JSON array with exactly 4 numbers between 0.0 and 1.0:
[LC_score, CC_score, FL_score, SA_score]

Your response should be a valid JSON object:
{{
  "{message_id}": [LC_score, CC_score, FL_score, SA_score]
}}
"""
    return prompt


def get_filler_token_batch_prompt(messages):
    """
    Batch evaluation with filler tokens to force independent consideration.
    Uses explicit thinking steps between each evaluation.
    """
    dimension_descriptions = get_dimension_descriptions()
    
    messages_text = "\n\n".join([f"Message {i+1}: \"{msg}\"" for i, msg in enumerate(messages)])
    
    prompt = f"""You are an expert language teacher evaluating tutoring messages.
{dimension_descriptions}

Here are the messages to evaluate:

{messages_text}

For each message, first think step by step about each dimension separately, then provide scores as a JSON array with exactly 4 numbers between 0.0 and 1.0:
[LC_score, CC_score, FL_score, SA_score]

Follow this format for each message:
THINKING ABOUT MESSAGE X:
- Lexical Complexity: [your analysis]
- Construction Complexity: [your analysis]
- Formality Level: [your analysis]
- Socratic Approach: [your analysis]
SCORES FOR MESSAGE X: [LC_score, CC_score, FL_score, SA_score]

After analyzing all messages individually, provide a final JSON object with all scores:
{{
  "1": [LC_score, CC_score, FL_score, SA_score],
  "2": [LC_score, CC_score, FL_score, SA_score],
  ...
}}
"""
    return prompt