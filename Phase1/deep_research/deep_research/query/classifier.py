"""
Query classifier module for deep research package.
Classifies queries to determine the optimal search strategy.
"""

import re
from collections import Counter

class QueryClassifier:
    """Classifies queries into different types for optimized search."""
    
    def __init__(self):
        """Initialize the query classifier."""
        # Define query type patterns
        self.patterns = {
            'factual': [
                r'\bwhat is\b', r'\bwho is\b', r'\bwhen was\b', r'\bwhere is\b',
                r'\bhow to\b', r'\bwhy does\b', r'\bdefinition of\b'
            ],
            'comparative': [
                r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference between\b',
                r'\bsimilarities\b', r'\bwhich is better\b'
            ],
            'opinion': [
                r'\bbest\b', r'\bworst\b', r'\brecommend\b', r'\breview\b',
                r'\bopinion\b', r'\bshould i\b'
            ],
            'research': [
                r'\bresearch\b', r'\bstudy\b', r'\banalysis\b', r'\bdata\b',
                r'\bstatistics\b', r'\bpaper\b', r'\bjournal\b'
            ],
            'news': [
                r'\bnews\b', r'\brecent\b', r'\blatest\b', r'\bupdate\b',
                r'\bcurrent events\b', r'\btoday\b', r'\bthis week\b'
            ]
        }
        
        # Keywords that suggest deep research is needed
        self.deep_research_keywords = {
            'comprehensive', 'detailed', 'thorough', 'in-depth',
            'analysis', 'investigate', 'research', 'examine'
        }
    
    def classify(self, parsed_query):
        """
        Classify the query into different types.
        
        Args:
            parsed_query (dict): The parsed query object
            
        Returns:
            dict: Classification results
        """
        query_text = parsed_query.get('cleaned_query', '').lower()
        
        # Check for each query type
        matches = {}
        for query_type, patterns in self.patterns.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, query_text, re.IGNORECASE):
                    match_count += 1
            
            if match_count > 0:
                matches[query_type] = match_count
        
        # Determine primary and secondary types
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'primary_type': sorted_matches[0][0] if sorted_matches else 'general',
            'secondary_types': [t for t, _ in sorted_matches[1:3]] if len(sorted_matches) > 1 else [],
            'confidence': sorted_matches[0][1] / len(self.patterns[sorted_matches[0][0]]) if sorted_matches else 0
        }
        
        # Check if deep research is required
        result['requires_deep_research'] = self._requires_deep_research(parsed_query)
        
        # Determine search strategy
        result['search_strategy'] = self._determine_search_strategy(result)
        
        return result
    
    def _requires_deep_research(self, parsed_query):
        """
        Determine if the query requires deep research.
        
        Args:
            parsed_query (dict): The parsed query
            
        Returns:
            bool: Whether deep research is required
        """
        # Check for deep research keywords
        query_text = parsed_query.get('cleaned_query', '').lower()
        for keyword in self.deep_research_keywords:
            if keyword in query_text:
                return True
        
        # Check query complexity
        keywords = parsed_query.get('keywords', [])
        if len(keywords) >= 4:  # Complex queries might need deeper research
            return True
        
        # Check for phrases (exact matches)
        phrases = parsed_query.get('phrases', [])
        if len(phrases) >= 2:  # Multiple exact phrases suggest specificity
            return True
        
        return False
    
    def _determine_search_strategy(self, classification):
        """
        Determine the optimal search strategy based on classification.
        
        Args:
            classification (dict): The query classification
            
        Returns:
            dict: Search strategy recommendations
        """
        primary_type = classification.get('primary_type')
        requires_deep = classification.get('requires_deep_research', False)
        
        strategy = {
            'search_depth': 'deep' if requires_deep else 'standard',
            'recommended_engines': [],
            'recommended_filters': []
        }
        
        # Customize strategy based on query type
        if primary_type == 'factual':
            strategy['recommended_engines'] = ['google', 'wikipedia']
            strategy['crawl_depth'] = 1
        
        elif primary_type == 'comparative':
            strategy['recommended_engines'] = ['google', 'bing']
            strategy['recommended_filters'] = ['comparison_sites']
            strategy['crawl_depth'] = 2
        
        elif primary_type == 'opinion':
            strategy['recommended_engines'] = ['reddit', 'quora', 'google']
            strategy['recommended_filters'] = ['forums', 'review_sites']
            strategy['crawl_depth'] = 2
        
        elif primary_type == 'research':
            strategy['recommended_engines'] = ['google_scholar', 'semantic_scholar', 'pubmed']
            strategy['recommended_filters'] = ['academic', 'pdf']
            strategy['crawl_depth'] = 3
        
        elif primary_type == 'news':
            strategy['recommended_engines'] = ['google_news', 'bing_news']
            strategy['recommended_filters'] = ['recent', 'news_sites']
            strategy['crawl_depth'] = 1
        
        else:  # general
            strategy['recommended_engines'] = ['google', 'bing', 'duckduckgo']
            strategy['crawl_depth'] = 2
        
        return strategy