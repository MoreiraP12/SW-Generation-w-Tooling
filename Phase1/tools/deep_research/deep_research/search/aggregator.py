"""
Search result aggregator for deep research package.
"""

from collections import defaultdict
from urllib.parse import urlparse

class SearchAggregator:
    """Aggregates search results from multiple engines."""
    
    def __init__(self, max_results=50, deduplication_threshold=0.8):
        """
        Initialize the search aggregator.
        
        Args:
            max_results (int): Maximum number of results to return
            deduplication_threshold (float): Threshold for considering results as duplicates
        """
        self.max_results = max_results
        self.deduplication_threshold = deduplication_threshold
        self.results = []
        self.seen_urls = set()
        self.seen_domains = defaultdict(int)
        self.source_weights = {
            'google': 1.0,
            'bing': 0.9,
            'duckduckgo': 0.85,
            'brave': 0.9,
            'default': 0.7
        }
    
    def add_results(self, new_results, source=None):
        """
        Add search results to the aggregator.
        
        Args:
            new_results (list): List of SearchResult objects
            source (str): Source of the results (engine name)
        """
        for result in new_results:
            # Skip if already seen this URL
            normalized_url = self._normalize_url(result.url)
            if normalized_url in self.seen_urls:
                continue
            
            # Mark as seen
            self.seen_urls.add(normalized_url)
            
            # Add to results
            self.results.append(result)
            
            # Update domain counter
            self.seen_domains[result.domain] += 1
    
    def get_results(self):
        """
        Get aggregated and ranked results.
        
        Returns:
            list: Ranked search results
        """
        # Rank results
        ranked_results = self._rank_results()
        
        # Return limited number of results
        return ranked_results[:self.max_results]
    
    def _normalize_url(self, url):
        """
        Normalize URL to avoid duplicates with minor differences.
        
        Args:
            url (str): URL to normalize
            
        Returns:
            str: Normalized URL
        """
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Normalize
            normalized = parsed.netloc + parsed.path.rstrip('/')
            
            # Remove common tracking parameters
            if parsed.query:
                params = parsed.query.split('&')
                filtered_params = []
                
                for param in params:
                    if param.startswith(('utm_', 'fbclid', 'gclid')):
                        continue
                    filtered_params.append(param)
                
                if filtered_params:
                    normalized += '?' + '&'.join(filtered_params)
            
            return normalized.lower()
            
        except Exception:
            return url.lower()
    
    def _rank_results(self):
        """
        Rank results based on relevance, source engine, domain diversity.
        
        Returns:
            list: Ranked search results
        """
        for result in self.results:
            # Base score is the reverse of rank (higher rank = lower score)
            base_score = 100 - min(result.rank, 100)
            
            # Source engine weight
            source_weight = self.source_weights.get(result.source_engine, self.source_weights['default'])
            
            # Domain diversity factor (penalize domains that appear too often)
            domain_count = self.seen_domains.get(result.domain, 0)
            diversity_factor = 1.0 / (1.0 + 0.1 * (domain_count - 1))
            
            # Calculate final score
            result.score = base_score * source_weight * diversity_factor
        
        # Sort by score
        ranked_results = sorted(self.results, key=lambda r: r.score, reverse=True)
        
        return ranked_results