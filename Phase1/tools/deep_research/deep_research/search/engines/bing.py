"""
Bing Search engine implementation.
"""

import os
import time
import json
import requests
from .base import SearchEngine, SearchResult
from datetime import datetime
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

class BingSearchEngine(SearchEngine):
    """Bing Search engine implementation using Bing Search API."""
    
    def __init__(self, api_key=None, rate_limit=3, timeout=10):
        """
        Initialize the Bing Search engine.
        
        Args:
            api_key (str): Bing Search API key
            rate_limit (int): Rate limit in requests per second
            timeout (int): Request timeout in seconds
        """
        super().__init__(name="bing", rate_limit=rate_limit, timeout=timeout)
        
        # Try to get API key from environment variables if not provided
        self.api_key = api_key or os.environ.get('BING_API_KEY')
        
        # Base URL for Bing Search API
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        
        # If API key is not available, use fallback method
        self.use_fallback = not self.api_key
        if self.use_fallback:
            print("Warning: Bing API key not found. Using fallback method.")
    
    def search(self, query, max_results=50, safe_search=True, **kwargs):
        """
        Perform a Bing search and return results.
        
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
    
    def _search_api(self, query, max_results=50, safe_search=True, **kwargs):
        """
        Search using the Bing Search API.
        
        Args:
            query (str): The prepared query string
            max_results (int): Maximum number of results to return
            safe_search (bool): Whether to enable safe search
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        results = []
        
        # Calculate number of API requests needed (each request can get up to 50 results)
        num_requests = min(max_results // 50 + (1 if max_results % 50 else 0), 2)
        
        for i in range(num_requests):
            # Set up parameters
            params = {
                'q': query,
                'count': min(50, max_results - i * 50),
                'offset': i * 50,
                'safeSearch': 'Strict' if safe_search else 'Off',
                'responseFilter': 'Webpages,News',
                'textDecorations': 'true',
                'textFormat': 'HTML'
            }
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Set up headers
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }
            
            # Make request
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse results
                webpages = data.get('webPages', {}).get('value', [])
                for rank, page in enumerate(webpages, start=i * 50 + 1):
                    result = SearchResult(
                        url=page.get('url', ''),
                        title=page.get('name', ''),
                        snippet=page.get('snippet', ''),
                        source_engine=self.name,
                        rank=rank,
                        domain=self._extract_domain(page.get('url', '')),
                        content_type=self._determine_content_type(page),
                        metadata={
                            'deep_links': page.get('deepLinks', []),
                            'date_last_crawled': page.get('dateLastCrawled', '')
                        }
                    )
                    results.append(result)
                
                # Respect rate limit
                time.sleep(1 / self.rate_limit)
                
            except Exception as e:
                print(f"Error searching Bing API: {e}")
                break
        
        return results
    
    def _search_fallback(self, query, max_results=50, safe_search=True, **kwargs):
        """
        Fallback search method when API key is not available.
        This implementation uses the Brave Search API, which has a free tier.
        
        Args:
            query (str): The prepared query string
            max_results (int): Maximum number of results to return
            safe_search (bool): Whether to enable safe search
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        results = []
        
        try:
            # Use Brave Search API (free tier available)
            api_key = os.environ.get('BRAVE_API_KEY')
            if not api_key:
                return []
                
            url = "https://api.search.brave.com/res/v1/web/search"
            
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": api_key
            }
            
            params = {
                "q": query,
                "count": min(max_results, 20),  # Brave API has a limit of 20 results per request
                "safesearch": safe_search
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse results
            web_results = data.get("web", {}).get("results", [])
            for rank, result_item in enumerate(web_results, start=1):
                result = SearchResult(
                    url=result_item.get("url", ""),
                    title=result_item.get("title", ""),
                    snippet=result_item.get("description", ""),
                    source_engine="brave",  # Mark the source as Brave since we're using their API
                    rank=rank,
                    domain=result_item.get("domain", ""),
                    content_type=self._determine_content_type_from_url(result_item.get("url", ""))
                )
                results.append(result)
                
        except Exception as e:
            print(f"Error in fallback search: {e}")
        
        return results
    
    def _extract_domain(self, url):
        """Extract domain from URL."""
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc
        except:
            return None
    
    def _determine_content_type(self, page):
        """Determine content type from page data."""
        url = page.get('url', '').lower()
        if url.endswith('.pdf'):
            return 'pdf'
        elif url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            return 'image'
        elif url.endswith(('.mp4', '.avi', '.mov', '.wmv')):
            return 'video'
        return 'webpage'
    
    def _determine_content_type_from_url(self, url):
        """Determine content type from URL."""
        url = url.lower()
        if url.endswith('.pdf'):
            return 'pdf'
        elif url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            return 'image'
        elif url.endswith(('.mp4', '.avi', '.mov', '.wmv')):
            return 'video'
        return 'webpage'