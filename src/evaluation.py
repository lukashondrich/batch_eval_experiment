"""
Functions to run the different evaluation approaches.
"""
import json
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
import yaml
from src.prompts import (
    get_baseline_batch_prompt,
    get_independent_prompt,
    get_filler_token_batch_prompt
)

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_sample_data(file_path="data/sample_data.yaml"):
    """Load sample data from YAML file."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def run_baseline_batch_evaluation(messages, model="gpt-4", temperature=0.0, repetition_penalty=0.0):
    """
    Run baseline batch evaluation approach.
    All messages evaluated at once.
    """
    prompt = get_baseline_batch_prompt(messages)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            frequency_penalty=repetition_penalty  
        )
        
        # Extract and parse the JSON response
        result_text = response.choices[0].message.content
        # Find JSON object in the response
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            results = json.loads(json_str)
        else:
            # Fallback parsing if clean JSON not found
            results = {"error": "Could not parse JSON response", "raw_response": result_text}
        
        return {
            "results": results,
            "prompt": prompt,
            "raw_response": result_text,
            "model": model,
            "temperature": temperature,
            "frequency_penalty": repetition_penalty,
            "method": "baseline_batch"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "frequency_penalty": repetition_penalty,
            "method": "baseline_batch"
        }

def run_independent_evaluations(messages, model="gpt-4", temperature=0.0, repetition_penalty=0.0):
    """
    Run independent evaluation approach.
    Each message evaluated separately.
    """
    all_results = {}
    prompts = {}
    raw_responses = {}
    
    for i, message in enumerate(messages):
        message_id = str(i + 1)
        prompt = get_independent_prompt(message, message_id)
        prompts[message_id] = prompt
        
        try:
            # Add delay to avoid rate limits
            if i > 0:
                time.sleep(1)
                
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                frequency_penalty=repetition_penalty 
            )
            
            result_text = response.choices[0].message.content
            raw_responses[message_id] = result_text
            
            # Parse JSON response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)
                # Merge the single result into all_results
                all_results.update(result)
            else:
                all_results[message_id] = {"error": "Could not parse JSON response"}
                
        except Exception as e:
            all_results[message_id] = {"error": str(e)}
    
    return {
        "results": all_results,
        "prompts": prompts,
        "raw_responses": raw_responses,
        "model": model,
        "temperature": temperature,
        "method": "independent"
    }

def run_filler_token_batch_evaluation(messages, model="gpt-4", temperature=0.0, repetition_penalty=0.0):
    """
    Run filler token batch evaluation approach.
    All messages at once but with forced thinking steps.
    """
    prompt = get_filler_token_batch_prompt(messages)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            frequency_penalty=repetition_penalty
        )
        result_text = response.choices[0].message.content
        
        # Extract the final JSON with scores
        json_start = result_text.rfind('{')
        json_end = result_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            results = json.loads(json_str)
        else:
            # Fallback parsing if clean JSON not found
            # Try to extract scores line by line
            results = {}
            for i in range(1, len(messages) + 1):
                score_marker = f"SCORES FOR MESSAGE {i}:"
                if score_marker in result_text:
                    # Find the scores after the marker
                    start_idx = result_text.find(score_marker) + len(score_marker)
                    end_idx = result_text.find("\n", start_idx)
                    if end_idx == -1:  # If no newline, go to the end
                        end_idx = len(result_text)
                    score_text = result_text[start_idx:end_idx].strip()
                    # Extract the array portion
                    array_start = score_text.find('[')
                    array_end = score_text.find(']') + 1
                    if array_start >= 0 and array_end > array_start:
                        try:
                            scores = json.loads(score_text[array_start:array_end])
                            results[str(i)] = scores
                        except:
                            results[str(i)] = {"error": "Could not parse score array"}
        
        return {
            "results": results,
            "prompt": prompt,
            "raw_response": result_text,
            "model": model,
            "temperature": temperature,
            "frequency_penalty": repetition_penalty,
            "method": "filler_token_batch"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "frequency_penalty": repetition_penalty,
            "method": "filler_token_batch"
        }

def run_all_evaluations(sample_data, trials=10, model="gpt-4", temperature=0.0, repetition_penalty=0.0):
    """
    Run all three evaluation approaches multiple times.
    
    Args:
        sample_data: Dictionary containing messages and ground truth scores
        trials: Number of trials to run for each method
        model: Model to use for evaluations
        temperature: Temperature setting for generation
        repetition_penalty: Controls repetition in generation
        
    Returns:
        Dictionary with results from all approaches and trials
    """
    # Extract messages from sample data
    messages = [sample["message"] for sample in sample_data["samples"]]
    
    results = {
        "baseline_batch": [],
        "independent": [],
        "filler_token_batch": [],
        "metadata": {
            "model": model,
            "temperature": temperature,
            "frequency_penalty": repetition_penalty,
            "trials": trials,
            "timestamp": time.time()
        },
        "ground_truth": {str(sample["id"]): sample["scores"] for sample in sample_data["samples"]}
    }
    
    # Run multiple trials for each approach
    for trial in range(trials):
        print(f"Running trial {trial+1}/{trials}...")
        
        print("  Baseline batch evaluation...")
        baseline_result = run_baseline_batch_evaluation(messages, model, temperature, repetition_penalty)
        results["baseline_batch"].append(baseline_result)
        
        print("  Independent evaluations...")
        independent_result = run_independent_evaluations(messages, model, temperature, repetition_penalty)
        results["independent"].append(independent_result)
        
        print("  Filler token batch evaluation...")
        filler_result = run_filler_token_batch_evaluation(messages, model, temperature, repetition_penalty)
        results["filler_token_batch"].append(filler_result)

    return results

def save_results(results, filename="data/raw/evaluation_results.json"):
    """Save evaluation results to a JSON file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename