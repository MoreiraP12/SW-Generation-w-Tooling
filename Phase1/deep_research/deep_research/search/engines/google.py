"""
Google Search engine implementation using SerpAPI.
"""

import os
import time
import json
import requests
from urllib.parse import urlencode
from .base import SearchEngine, SearchResult
from datetime import datetime, timedelta

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

class GoogleSearchEngine(SearchEngine):
    """Google Search engine implementation using Custom Google Search JSON API."""
    
    def __init__(self, api_key=None, cx=None, rate_limit=5, timeout=10):
        """
        Initialize the Google Search engine.
        
        Args:
            api_key (str): Google Custom Search API key
            cx (str): Google Custom Search Engine ID
            rate_limit (int): Rate limit in requests per second
            timeout (int): Request timeout in seconds
        """
        super().__init__(name="google", rate_limit=rate_limit, timeout=timeout)
        
        # Try to get API key from environment variables if not provided
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        self.cx = cx or os.environ.get('GOOGLE_CX')
        
        # Base URL for Google Custom Search
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # If API key and CX are not available, use fallback method
        self.use_fallback = not (self.api_key and self.cx)
        if self.use_fallback:
            print("Warning: Google API key or CX not found. Using fallback method.")
    
    def search(self, query, max_results=10, safe_search=True, **kwargs):
        """
        Perform a Google search and return results.
        
        Args:
            query (str or dict): The query string or parsed query object
            max_results (int): Maximum number of results to return
            safe_search (bool): Whether to enable safe search
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        # Prepare query
        prepared_query = self._prepare_query(query)
        
        # Use appropriate search method
        if self.use_fallback:
            return self._search_fallback(prepared_query, max_results, safe_search, **kwargs)
        else:
            return self._search_api(prepared_query, max_results, safe_search, **kwargs)
    
    def _search_api(self, query, max_results=10, safe_search=True, **kwargs):
        """
        Search using the Google Custom Search API.
        
        Args:
            query (str): The prepared query string
            max_results (int): Maximum number of results to return
            safe_search (bool): Whether to enable safe search
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        results = []
        
        # Calculate number of API requests needed (each request can get up to 10 results)
        num_requests = min(max_results // 10 + (1 if max_results % 10 else 0), 10)
        
        for i in range(num_requests):
            # Set up parameters
            params = {
                'key': self.api_key,
                'cx': self.cx,
                'q': query,
                'num': min(10, max_results - i * 10),
                'start': i * 10 + 1,
                'safe': 'active' if safe_search else 'off',
            }
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Make request
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse results
                items = data.get('items', [])
                for rank, item in enumerate(items, start=i * 10 + 1):
                    result = SearchResult(
                        url=item.get('link', ''),
                        title=item.get('title', ''),
                        snippet=item.get('snippet', ''),
                        source_engine=self.name,
                        rank=rank,
                        domain=item.get('displayLink', None),
                        content_type=self._determine_content_type(item),
                        metadata={
                            'page_map': item.get('pagemap', {}),
                            'mime_type': item.get('mime', '')
                        }
                    )
                    results.append(result)
                
                # Respect rate limit
                time.sleep(1 / self.rate_limit)
                
            except Exception as e:
                print(f"Error searching Google API: {e}")
                break
        
        return results
    
    def _search_fallback(self, query, max_results=10, safe_search=True, **kwargs):
        """
        Fallback search method using DuckDuckGo or another free alternative.
        
        Args:
            query (str): The prepared query string
            max_results (int): Maximum number of results to return
            safe_search (bool): Whether to enable safe search
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        # In a real implementation, you would use a free alternative here
        # For this example, we'll use a simplified approach to demonstrate
        # A better alternative would be to implement a proper scraper or use another free API
        
        results = []
        try:
            # Use rapidapi.com search API (has some free tier)
            # You would need to sign up for a free tier API key
            api_key = os.environ.get('RAPIDAPI_KEY')
            if not api_key:
                return []
                
            url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/WebSearchAPI"
            
            params = {
                "q": query,
                "pageNumber": 1,
                "pageSize": max_results,
                "autoCorrect": "true",
                "safeSearch": str(safe_search).lower()
            }
            
            headers = {
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
            }
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            # Parse results
            for rank, item in enumerate(data.get("value", []), start=1):
                result = SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                    source_engine=self.name,
                    rank=rank,
                    date=self._parse_date(item.get("datePublished")),
                    domain=item.get("provider", {}).get("name", None),
                )
                results.append(result)
                
        except Exception as e:
            print(f"Error in fallback search: {e}")
        
        return results
    
    def _determine_content_type(self, item):
        """Determine content type from item data."""
        mime = item.get('mime', '')
        if mime == 'application/pdf':
            return 'pdf'
        elif mime.startswith('image/'):
            return 'image'
        elif mime.startswith('video/'):
            return 'video'
        return 'webpage'
    
    def _parse_date(self, date_str):
        """Parse date string to datetime object."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None