"""
Load and render prompts from YAML file using Jinja2.
"""
import yaml
from jinja2 import Environment
from pathlib import Path

# Load prompts YAML once
def load_prompts_yaml():
    """
    Load prompts.yaml from the project-level data directory.
    """
    # project root is one level above src/
    base_dir = Path(__file__).resolve().parents[1]
    path = base_dir / 'data' / 'prompts.yaml'
    if not path.is_file():
        raise FileNotFoundError(f"Could not find prompts.yaml at {path}")
    with path.open() as f:
        return yaml.safe_load(f)

_prompts = load_prompts_yaml()

# Initialize Jinja2 environment for string templates
env = Environment()

def _render(template_key: str, **kwargs):
    """
    Render the specified template with provided kwargs.
    """
    if template_key not in _prompts:
        raise KeyError(f"Prompt '{template_key}' not found in prompts.yaml")
    template_str = _prompts[template_key]
    template = env.from_string(template_str)
    
    # Always pass dimension_descriptions
    kwargs['dimension_descriptions'] = _prompts['dimension_descriptions']
    
    return template.render(**kwargs)

def get_baseline_batch_prompt(messages):
    """Get the basic batch evaluation prompt."""
    return _render('basic_batch', messages=[{'index': i+1, 'text': msg} for i, msg in enumerate(messages)])

def get_filler_token_batch_prompt(messages):
    """Get the sophisticated batch evaluation prompt with step-by-step reasoning."""
    return _render('sophisticated_batch', messages=[{'index': i+1, 'text': msg} for i, msg in enumerate(messages)])

def get_independent_prompt(message, message_id):
    """
    Independent evaluation prompt.
    One message at a time.
    """
    return _render('independent', message=message, message_id=message_id)