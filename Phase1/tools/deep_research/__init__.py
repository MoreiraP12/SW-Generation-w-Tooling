"""
Deep Research - A Python package for conducting comprehensive online research.

This package provides tools to query multiple search engines, crawl websites,
extract relevant information, and synthesize answers based on the gathered data.
"""

__version__ = "0.1.0"

# Import key components for easier access
from deep_research.query import process_query
from deep_research.search import search_multiple_engines
from deep_research.crawler import crawl_website
from deep_research.extractor import extract_content
from deep_research.analysis import analyze_content
from deep_research.synthesis import synthesize_answer

# Main function that orchestrates the entire research process
def research(query, depth="standard", max_sources=10, timeout=60):
    """
    Conduct deep research based on a query.
    
    Args:
        query (str): The research query
        depth (str): Research depth ("quick", "standard", or "deep")
        max_sources (int): Maximum number of sources to analyze
        timeout (int): Maximum time in seconds for the research
        
    Returns:
        dict: Research results with synthesized answer and sources
    """
    # Process the query
    query_data = process_query(query)
    
    # Search for relevant sources
    search_results = search_multiple_engines(
        query_data.get('parsed_query'),
        query_data.get('expanded_queries', []),
        strategy=query_data.get('query_type', {}).get('search_strategy', {})
    )
    
    # Crawl the top results
    crawl_depth = 1
    if depth == "standard":
        crawl_depth = 2
    elif depth == "deep":
        crawl_depth = 3
        
    crawled_data = []
    for result in search_results[:max_sources]:
        content = crawl_website(result['url'], depth=crawl_depth)
        if content:
            crawled_data.append({
                'url': result['url'],
                'title': result['title'],
                'content': content
            })
    
    # Extract relevant content
    extracted_data = []
    for item in crawled_data:
        extracted = extract_content(item['content'], query_data['parsed_query'])
        if extracted:
            extracted_data.append({
                'url': item['url'],
                'title': item['title'],
                'extracted_content': extracted
            })
    
    # Analyze the extracted content
    analyzed_data = analyze_content(extracted_data, query_data['parsed_query'])
    
    # Synthesize the final answer
    answer = synthesize_answer(analyzed_data, query_data)
    
    return {
        'query': query,
        'answer': answer.get('synthesized_answer', ''),
        'confidence': answer.get('confidence', 0),
        'sources': answer.get('sources', []),
        'query_analysis': query_data
    }

# Export the main research function as the default API
__all__ = [
    'research', 'process_query', 'search_multiple_engines', 
    'crawl_website', 'extract_content', 'analyze_content', 
    'synthesize_answer', '__version__'
]