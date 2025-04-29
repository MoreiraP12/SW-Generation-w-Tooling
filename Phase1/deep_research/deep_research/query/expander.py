"""
Query expander module for deep research package.
Expands queries with synonyms, related terms, etc.
"""

import requests
from nltk.corpus import wordnet as wn

class QueryExpander:
    """Expands queries with synonyms and related terms."""
    
    def __init__(self, max_expansions=3, use_api=False, api_key=None):
        """
        Initialize the query expander.
        
        Args:
            max_expansions (int): Maximum number of expanded queries to generate
            use_api (bool): Whether to use external API for better expansions
            api_key (str): API key for external service if used
        """
        self.max_expansions = max_expansions
        self.use_api = use_api
        self.api_key = api_key
    
    def expand(self, parsed_query):
        """
        Expand the parsed query with related terms.
        
        Args:
            parsed_query (dict): The parsed query object
            
        Returns:
            list: List of expanded query strings
        """
        # Get the original keywords
        keywords = parsed_query.get('keywords', [])
        
        # Get expansions for each keyword
        expanded_terms = {}
        for keyword in keywords:
            expanded_terms[keyword] = self._get_term_expansions(keyword)
        
        # Generate expanded queries
        expanded_queries = self._generate_expanded_queries(parsed_query, expanded_terms)
        
        # Limit the number of expansions
        return expanded_queries[:self.max_expansions]
    
    def _get_term_expansions(self, term):
        """
        Get expansions for a single term.
        
        Args:
            term (str): The term to expand
            
        Returns:
            list: List of related terms
        """
        expansions = []
        
        # Use WordNet for synonyms
        for synset in wn.synsets(term):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != term and synonym not in expansions:
                    expansions.append(synonym)
        
        # Use external API if configured
        if self.use_api and self.api_key:
            api_expansions = self._get_api_expansions(term)
            for exp in api_expansions:
                if exp not in expansions:
                    expansions.append(exp)
        
        return expansions[:5]  # Limit to top 5 expansions per term
    
    def _get_api_expansions(self, term):
        """
        Get expansions from external API (placeholder).
        
        Args:
            term (str): The term to expand
            
        Returns:
            list: List of related terms from API
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Call an external API like Google Knowledge Graph, DataMuse, etc.
        # 2. Process the response to extract related terms
        # Example with DataMuse (doesn't require API key):
        try:
            response = requests.get(f"https://api.datamuse.com/words?ml={term}&max=5")
            if response.status_code == 200:
                data = response.json()
                return [item['word'] for item in data]
        except Exception:
            pass
        
        return []
    
    def _generate_expanded_queries(self, parsed_query, expanded_terms):
        """
        Generate expanded queries by substituting terms.
        
        Args:
            parsed_query (dict): The parsed query
            expanded_terms (dict): Dictionary of terms and their expansions
            
        Returns:
            list: List of expanded query strings
        """
        original_query = parsed_query.get('cleaned_query', '')
        expanded_queries = [original_query]
        
        # Simple expansion strategy: replace one keyword at a time
        for keyword, expansions in expanded_terms.items():
            for expansion in expansions:
                # Create a new query with this term replaced
                new_query = original_query.replace(keyword, expansion)
                if new_query != original_query and new_query not in expanded_queries:
                    expanded_queries.append(new_query)
        
        return expanded_queries