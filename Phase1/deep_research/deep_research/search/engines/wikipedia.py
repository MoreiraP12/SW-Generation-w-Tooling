"""
Wikipedia Search engine implementation.
"""

import requests
import re
import html
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
from .base import SearchEngine, SearchResult
from datetime import datetime

class WikipediaSearchEngine(SearchEngine):
    """Wikipedia Search engine implementation using Wikipedia's API."""
    
    def __init__(self, rate_limit=1, timeout=10, language="en"):
        """
        Initialize the Wikipedia Search engine.
        
        Args:
            rate_limit (int): Rate limit in requests per second
            timeout (int): Request timeout in seconds
            language (str): Wikipedia language code (default: "en" for English)
        """
        super().__init__(name="wikipedia", rate_limit=rate_limit, timeout=timeout)
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.article_base_url = f"https://{language}.wikipedia.org/wiki/"
    
    def search(self, query, max_results=10, safe_search=True, **kwargs):
        """
        Perform a Wikipedia search and return results.
        
        Args:
            query (str or dict): The query string or parsed query object
            max_results (int): Maximum number of results to return
            safe_search (bool): Whether to enable safe search (not used for Wikipedia)
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        # Prepare query
        prepared_query = self._prepare_query(query)
        
        # Perform search
        return self._search_api(prepared_query, max_results, **kwargs)
    
    def _search_api(self, query, max_results=10, **kwargs):
        """
        Search using the Wikipedia API.
        
        Args:
            query (str): The prepared query string
            max_results (int): Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            list: Search results
        """
        results = []
        
        try:
            # Set up parameters for search
            search_params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': min(max_results, 50),  # Wikipedia API limit is 50
                'srinfo': 'suggestion',
                'srprop': 'snippet|titlesnippet|sectiontitle|sectionsnippet|categorysnippet|redirecttitle|redirectsnippet',
                'format': 'json',
                'utf8': 1,
            }
            
            # Add any additional parameters
            search_params.update(kwargs.get('search_params', {}))
            
            # Make request
            response = self.session.get(
                self.base_url,
                params=search_params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Process search results
            if 'query' in data and 'search' in data['query']:
                search_results = data['query']['search']
                
                for rank, result in enumerate(search_results, 1):
                    if rank > max_results:
                        break
                    
                    page_id = result.get('pageid')
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    
                    # Clean up HTML in snippet
                    snippet = self._clean_html(snippet)
                    
                    # Create article URL
                    url = f"{self.article_base_url}{quote(title.replace(' ', '_'))}"
                    
                    # Get extract for better content
                    extract = self._get_article_extract(page_id) if page_id else ""
                    
                    # Create search result
                    search_result = SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet if not extract else extract[:200] + "...",
                        source_engine=self.name,
                        rank=rank,
                        domain=f"{self.language}.wikipedia.org",
                        content_type="webpage",
                        metadata={
                            'page_id': page_id,
                            'full_extract': extract
                        }
                    )
                    results.append(search_result)
            
        except Exception as e:
            print(f"Error searching Wikipedia: {e}")
        
        return results
    
    def _get_article_extract(self, page_id, chars=500):
        """
        Get an extract (summary) of a Wikipedia article.
        
        Args:
            page_id (int): Wikipedia page ID
            chars (int): Maximum number of characters for the extract
            
        Returns:
            str: Article extract
        """
        try:
            extract_params = {
                'action': 'query',
                'pageids': page_id,
                'prop': 'extracts',
                'exintro': 1,          # Only get content from intro section
                'explaintext': 1,      # Return plain text instead of HTML
                'exsectionformat': 'plain',
                'exchars': chars,
                'format': 'json',
                'utf8': 1,
            }
            
            response = self.session.get(
                self.base_url,
                params=extract_params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                if str(page_id) in data['query']['pages']:
                    page_data = data['query']['pages'][str(page_id)]
                    extract = page_data.get('extract', '')
                    return extract
            
        except Exception as e:
            print(f"Error getting Wikipedia extract: {e}")
        
        return ""
    
    def get_full_article(self, page_id=None, title=None):
        """
        Get the full content of a Wikipedia article.
        
        Args:
            page_id (int): Wikipedia page ID (preferred)
            title (str): Article title (used if page_id is None)
            
        Returns:
            dict: Article content
        """
        if not page_id and not title:
            return None
        
        try:
            params = {
                'action': 'query',
                'prop': 'extracts|categories|links|images|info',
                'inprop': 'url',
                'format': 'json',
                'utf8': 1,
            }
            
            if page_id:
                params['pageids'] = page_id
            elif title:
                params['titles'] = title
            
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error getting full Wikipedia article: {e}")
            return None
    
    def _clean_html(self, html_text):
        """
        Clean HTML text.
        
        Args:
            html_text (str): HTML text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove HTML tags
        soup = BeautifulSoup(html_text, 'html.parser')
        text = soup.get_text()
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_synthesis(self, query, max_results=3):
        """
        Get a synthesis of information from Wikipedia based on a query.
        
        Args:
            query (str): The query string
            max_results (int): Maximum number of results to consider
            
        Returns:
            dict: Synthesized information
        """
        search_results = self._search_api(query, max_results=max_results)
        
        if not search_results:
            return {
                'success': False,
                'message': 'No results found',
                'query': query
            }
        
        # Use the top result for synthesis
        top_result = search_results[0]
        page_id = top_result.metadata.get('page_id')
        
        if not page_id:
            return {
                'success': False,
                'message': 'No page ID found for top result',
                'results': search_results
            }
        
        # Get full article data
        article_data = self.get_full_article(page_id=page_id)
        
        if not article_data or 'query' not in article_data or 'pages' not in article_data['query']:
            return {
                'success': False,
                'message': 'Could not retrieve article data',
                'results': search_results
            }
        
        # Process article data
        page_data = article_data['query']['pages'][str(page_id)]
        
        return {
            'success': True,
            'query': query,
            'title': page_data.get('title', ''),
            'url': top_result.url,
            'extract': page_data.get('extract', ''),
            'categories': [cat.get('title', '').replace('Category:', '') 
                          for cat in page_data.get('categories', [])],
            'related_topics': [link.get('title', '') 
                              for link in page_data.get('links', [])[:5]],
            'all_results': search_results
        }