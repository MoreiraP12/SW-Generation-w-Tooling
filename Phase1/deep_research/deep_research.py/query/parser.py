"""
Query parser module for deep research package.
Handles parsing raw query text into structured format.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class QueryParser:
    """Parser for converting raw query text into structured format."""
    
    def __init__(self):
        """Initialize the query parser."""
        # Download necessary NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def parse(self, query_text):
        """
        Parse the raw query text.
        
        Args:
            query_text (str): The raw query text
            
        Returns:
            dict: Structured representation of the query
        """
        # Clean query
        cleaned_query = self._clean_query(query_text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_query)
        
        # Remove stop words
        keywords = [token.lower() for token in tokens if token.lower() not in self.stop_words and token.isalnum()]
        
        # Extract special directives (e.g., site:example.com)
        directives = self._extract_directives(query_text)
        
        # Handle quotes (exact phrases)
        phrases = self._extract_phrases(query_text)
        
        return {
            'cleaned_query': cleaned_query,
            'tokens': tokens,
            'keywords': keywords,
            'directives': directives,
            'phrases': phrases
        }
    
    def _clean_query(self, query_text):
        """Clean the query text by removing extra whitespace."""
        return ' '.join(query_text.strip().split())
    
    def _extract_directives(self, query_text):
        """
        Extract search directives like site:example.com.
        
        Args:
            query_text (str): The raw query text
            
        Returns:
            dict: Extracted directives
        """
        directives = {}
        
        # Extract site directive
        site_match = re.search(r'site:(\S+)', query_text)
        if site_match:
            directives['site'] = site_match.group(1)
        
        # Extract filetype directive
        filetype_match = re.search(r'filetype:(\S+)', query_text)
        if filetype_match:
            directives['filetype'] = filetype_match.group(1)
        
        # Extract date range directive
        date_range_match = re.search(r'daterange:(\S+)', query_text)
        if date_range_match:
            directives['daterange'] = date_range_match.group(1)
        
        return directives
    
    def _extract_phrases(self, query_text):
        """
        Extract exact phrases (quoted text) from the query.
        
        Args:
            query_text (str): The raw query text
            
        Returns:
            list: Exact phrases to search for
        """
        phrases = []
        pattern = r'"([^"]*)"'
        matches = re.findall(pattern, query_text)
        
        for match in matches:
            if match.strip():
                phrases.append(match.strip())
        
        return phrases