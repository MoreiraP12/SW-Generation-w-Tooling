"""
Search module for deep research package.
This module handles searching across multiple search engines.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .engines import GoogleSearchEngine, BingSearchEngine, DuckDuckGoSearchEngine, WikipediaSearchEngine
from .aggregator import SearchAggregator

__all__ = ['search', 'SearchAggregator', 'GoogleSearchEngine', 'BingSearchEngine', 'DuckDuckGoSearchEngine', 'WikipediaSearchEngine']

def search(query, engines=None, max_results=30, safe_search=True):
    """
    Perform a search across multiple search engines.
    
    Args:
        query (str or dict): The query string or parsed query object
        engines (list): List of search engine names to use (default: all available)
        max_results (int): Maximum number of results to return
        safe_search (bool): Whether to enable safe search
        
    Returns:
        list: Aggregated search results
    """
    # Create aggregator
    aggregator = SearchAggregator(max_results=max_results)
    
    # Determine which engines to use
    available_engines = {
        'google': GoogleSearchEngine(),
        'bing': BingSearchEngine(),
        'duckduckgo': DuckDuckGoSearchEngine(),
        'wikipedia': WikipediaSearchEngine()
    }
    
    if engines is None:
        engines = list(available_engines.keys())
    
    # Execute search on each engine
    for engine_name in engines:
        if engine_name in available_engines:
            engine = available_engines[engine_name]
            results = engine.search(query, safe_search=safe_search)
            aggregator.add_results(results, source=engine_name)
    
    # Return aggregated and ranked results
    return aggregator.get_results()