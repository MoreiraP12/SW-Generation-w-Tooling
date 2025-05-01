"""
Base classes for search engines.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import quote_plus

@dataclass
class SearchResult:
    """Class representing a search result."""
    
    url: str
    title: str
    snippet: str
    source_engine: str
    rank: int = 0
    date: Optional[datetime] = None
    domain: Optional[str] = None
    content_type: str = "webpage"
    is_pdf: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize additional fields after creation."""
        from urllib.parse import urlparse
        
        # Extract domain from URL if not provided
        if self.domain is None:
            parsed_url = urlparse(self.url)
            self.domain = parsed_url.netloc
        
        # Initialize metadata dictionary if None
        if self.metadata is None:
            self.metadata = {}
        
        # Check if URL points to a PDF
        if self.url.lower().endswith('.pdf'):
            self.is_pdf = True
            self.content_type = "pdf"

class SearchEngine:
    """Base class for search engines."""
    
    def __init__(self, name="generic", rate_limit=1, timeout=10):
        """
        Initialize the search engine.
        
        Args:
            name (str): Name of the search engine
            rate_limit (int): Rate limit in requests per second
            timeout (int): Request timeout in seconds
        """
        self.name = name
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search(self, query, max_results=10, safe_search=True, **kwargs):
        """
        Perform a search and return results.
        
        Args:
            query (str or dict): The query string or parsed query object
            max_results (int): Maximum number of results to return
            safe_search (bool): Whether to enable safe search
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        raise NotImplementedError("Subclasses must implement search method")
    
    def _prepare_query(self, query):
        """
        Prepare query for searching.
        
        Args:
            query (str or dict): The query string or parsed query object
            
        Returns:
            str: Prepared query string
        """
        if isinstance(query, dict):
            # Handle parsed query object
            cleaned_query = query.get('cleaned_query', '')
            phrases = query.get('phrases', [])
            directives = query.get('directives', {})
            
            # Add phrases in quotes
            for phrase in phrases:
                if f'"{phrase}"' not in cleaned_query:
                    cleaned_query += f' "{phrase}"'
            
            # Add directives
            for directive, value in directives.items():
                directive_str = f"{directive}:{value}"
                if directive_str not in cleaned_query:
                    cleaned_query += f" {directive_str}"
            
            return cleaned_query.strip()
        
        # If query is a string, return as is
        return query
    
    def _encode_query(self, query):
        """
        URL encode the query.
        
        Args:
            query (str): The query string
            
        Returns:
            str: URL encoded query
        """
        return quote_plus(query)