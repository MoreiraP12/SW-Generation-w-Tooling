"""
Search engines for deep research package.
"""

from .base import SearchEngine, SearchResult
from .google import GoogleSearchEngine
from .bing import BingSearchEngine
from .duckduckgo import DuckDuckGoSearchEngine
from .wikipedia import WikipediaSearchEngine

__all__ = ['SearchEngine', 'SearchResult', 'GoogleSearchEngine', 'BingSearchEngine', 'DuckDuckGoSearchEngine', 'WikipediaSearchEngine']