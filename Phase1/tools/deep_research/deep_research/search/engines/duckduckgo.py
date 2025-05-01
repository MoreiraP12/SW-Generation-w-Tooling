"""
Updated DuckDuckGo Search engine implementation with Abstract feature.
"""

import requests
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote, quote_plus
from .base import SearchEngine, SearchResult

class DuckDuckGoSearchEngine(SearchEngine):
    """DuckDuckGo Search engine implementation that scrapes search results and includes Instant Answer abstracts."""
    
    def __init__(self, rate_limit=1, timeout=10):
        """Initialize the DuckDuckGo Search engine."""
        super().__init__(name="duckduckgo", rate_limit=rate_limit, timeout=timeout)
        self.search_url = "https://duckduckgo.com/html/"
        self.instant_answer_url = "https://api.duckduckgo.com/"
    
    def search(self, query, max_results=30, safe_search=True, **kwargs):
        """Perform a DuckDuckGo search and return results."""
        # Prepare query
        prepared_query = self._prepare_query(query)
        
        # Get instant answer abstract if available
        abstract_data = self._get_instant_answer(prepared_query)
        
        # Get regular search results
        search_results = self._search_html(prepared_query, max_results, safe_search, **kwargs)
        
        # Add abstract as first result if available
        if abstract_data and abstract_data.get('Abstract'):
            # Create a special result for the abstract
            abstract_result = SearchResult(
                url=abstract_data.get('AbstractURL', ''),
                title=f"[Abstract] {abstract_data.get('Heading', prepared_query)}",
                snippet=abstract_data.get('Abstract', ''),
                source_engine=self.name,
                rank=0,  # Give it top rank
                domain=abstract_data.get('AbstractSource', 'DuckDuckGo'),
                content_type='abstract',  # Special content type for abstracts
                metadata={
                    'is_abstract': True,
                    'abstract_source': abstract_data.get('AbstractSource', ''),
                    'related_topics': len(abstract_data.get('RelatedTopics', [])),
                    'definition': abstract_data.get('Definition', ''),
                    'definition_source': abstract_data.get('DefinitionSource', '')
                }
            )
            # Insert at beginning of results
            search_results.insert(0, abstract_result)
        
        return search_results
    
    def _get_instant_answer(self, query):
        """
        Get instant answer data from DuckDuckGo API.
        
        Args:
            query (str): The query string
            
        Returns:
            dict: Instant answer data or None if not available
        """
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                't': 'deepresearch'  # App name
            }
            
            response = self.session.get(
                self.instant_answer_url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                # Only return if there's meaningful data
                if data.get('Abstract') or data.get('Definition') or data.get('RelatedTopics'):
                    return data
            return None
        except Exception as e:
            print(f"Error getting DuckDuckGo instant answer: {e}")
            return None
    
    def _search_html(self, query, max_results=30, safe_search=True, **kwargs):
        """Search by scraping the DuckDuckGo HTML search page."""
        results = []
        
        try:
            # Set up parameters
            params = {
                'q': query,
                'kl': 'us-en',  # Locale
                'kp': '1' if safe_search else '-1',  # Safe search
            }
            
            # Custom headers to simulate a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://duckduckgo.com/',
            }
            
            # Make the request
            response = self.session.get(
                self.search_url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse the response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract results
            result_elements = soup.select('.result')
            
            for rank, result_element in enumerate(result_elements, 1):
                if rank > max_results:
                    break
                
                # Skip ad results
                if 'result--ad' in result_element.get('class', []):
                    continue
                
                # Extract result details
                title_element = result_element.select_one('.result__title')
                snippet_element = result_element.select_one('.result__snippet')
                url_element = result_element.select_one('.result__url')
                
                if title_element:
                    # Get the title
                    title = title_element.get_text(strip=True)
                    
                    # Get the URL
                    link_element = title_element.select_one('a')
                    href = link_element['href'] if link_element else ''
                    url = self._extract_redirect_url(href)
                    
                    # Get the snippet
                    snippet = snippet_element.get_text(strip=True) if snippet_element else ""
                    
                    # Get the domain
                    domain = url_element.get_text(strip=True) if url_element else self._extract_domain(url)
                    
                    # Create result
                    result = SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        source_engine=self.name,
                        rank=rank,
                        domain=domain,
                        content_type=self._determine_content_type(url)
                    )
                    results.append(result)
            
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
        
        return results
    
    def _extract_redirect_url(self, href):
        """Extract the actual URL from DuckDuckGo's redirect URL."""
        try:
            # DuckDuckGo uses /l/?kh=-1&uddg=URL format
            if '/l/?kh=' in href:
                match = re.search(r'uddg=([^&]+)', href)
                if match:
                    return unquote(match.group(1))
            return href
        except Exception:
            return href
    
    def _extract_domain(self, url):
        """Extract domain from URL."""
        try:
            return urlparse(url).netloc
        except:
            return None
    
    def _determine_content_type(self, url):
        """Determine content type from URL."""
        url = url.lower() if url else ""
        if url.endswith('.pdf'):
            return 'pdf'
        elif url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            return 'image'
        elif url.endswith(('.mp4', '.avi', '.mov', '.wmv')):
            return 'video'
        return 'webpage'