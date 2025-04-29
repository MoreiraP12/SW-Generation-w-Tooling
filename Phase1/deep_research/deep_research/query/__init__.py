"""
Query processing module for deep research package.
This module handles query parsing, expansion, and classification.
"""

from .parser import QueryParser
from .expander import QueryExpander
from .classifier import QueryClassifier

__all__ = ['QueryParser', 'QueryExpander', 'QueryClassifier', 'process_query']


def process_query(query_text, expand=True, classify=True):
    """
    Process a query through the entire pipeline.
    
    Args:
        query_text (str): The raw query text
        expand (bool): Whether to expand the query
        classify (bool): Whether to classify the query
        
    Returns:
        dict: A dictionary containing parsed query, expansions, and classification
    """
    parser = QueryParser()
    parsed_query = parser.parse(query_text)
    
    result = {
        'original_query': query_text,
        'parsed_query': parsed_query
    }
    
    if expand:
        expander = QueryExpander()
        expanded_queries = expander.expand(parsed_query)
        result['expanded_queries'] = expanded_queries
    
    if classify:
        classifier = QueryClassifier()
        query_type = classifier.classify(parsed_query)
        result['query_type'] = query_type
    
    return result