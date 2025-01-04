"""
Utility functions for AI model interactions and response handling.
This module manages API interactions with various AI models and provides
helper functions for response processing and leaderboard calculations.
"""

import os
import random
import logging
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from anthropic import Anthropic

# Configure logging with timestamp and log level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize API clients with error handling
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # Initialize DeepSeek client with custom base URL
    deepseek_client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
except Exception as e:
    logger.error(f"Failed to initialize API clients: {e}")
    raise

def get_chatgpt_response(prompt: str) -> str:
    """
    Get response from ChatGPT API.

    Args:
        prompt (str): The input text to send to the model

    Returns:
        str: The model's response text

    Raises:
        Exception: If API call fails
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error getting ChatGPT response: {str(e)}"

def get_claude_response(prompt: str) -> str:
    """
    Get response from Claude API.

    Args:
        prompt (str): The input text to send to the model

    Returns:
        str: The model's response text

    Raises:
        Exception: If API call fails
    """
    try:
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        return f"Error getting Claude response: {str(e)}"

def get_deepseek_response(prompt: str) -> str:
    """
    Get response from DeepSeek API.

    Args:
        prompt (str): The input text to send to the model

    Returns:
        str: The model's response text

    Raises:
        Exception: If API call fails
    """
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek API error: {str(e)}")
        return f"Error getting DeepSeek response: {str(e)}"

def get_responses(prompt: str) -> Dict[str, str]:
    """
    Generate responses from each AI model.

    Args:
        prompt (str): The input text to send to all models

    Returns:
        Dict[str, str]: Dictionary mapping model names to their responses
    """
    responses = {
        'ChatGPT 4.0': get_chatgpt_response(prompt),
        'Claude 3.5 Sonnet': get_claude_response(prompt),
        'DeepSeek v3': get_deepseek_response(prompt)
    }
    return responses

def shuffle_responses(responses: Dict[str, str]) -> Tuple[List[str], Dict[int, str]]:
    """
    Anonymize responses by shuffling them randomly.

    Args:
        responses (Dict[str, str]): Dictionary of model responses

    Returns:
        Tuple[List[str], Dict[int, str]]: Tuple containing:
            - List of shuffled response texts
            - Mapping of indices to original model names
    """
    # Extract models and outputs
    models = list(responses.keys())
    outputs = list(responses.values())

    # Create and shuffle indices
    shuffled_indices = list(range(len(models)))
    random.shuffle(shuffled_indices)

    # Create shuffled outputs and mapping
    shuffled_outputs = [outputs[i] for i in shuffled_indices]
    mapping = {i: models[shuffled_indices[i]] for i in range(len(models))}

    # Reset random seed for extra randomization
    random.seed()

    return shuffled_outputs, mapping

def calculate_leaderboard(votes: List) -> Dict[str, int]:
    """
    Calculate the current standings from vote history.

    Args:
        votes (List): List of Vote objects from database

    Returns:
        Dict[str, int]: Dictionary mapping model names to their vote counts
    """
    # Initialize standings with zero counts for all models
    standings = {
        'ChatGPT 4.0': 0,
        'Claude 3.5 Sonnet': 0,
        'DeepSeek v3': 0
    }

    # Count votes for each model
    for vote in votes:
        if vote.winner in standings:
            standings[vote.winner] += 1

    return standings