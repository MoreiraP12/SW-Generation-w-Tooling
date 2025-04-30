"""
Web crawler module with Hugging Face model integration for deep research.
"""

import requests
import time
import random
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import os
import re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging
import hashlib
import json
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deep_research.crawler')


class ModelHandler:
    """Handles loading and inference for Hugging Face models."""
    
    def __init__(self, summarization_model="facebook/bart-large-cnn", 
                 relevance_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 qa_model="deepset/roberta-base-squad2",
                 use_gpu=torch.cuda.is_available()):
        """Initialize model handler with specified models."""
        self.device = 0 if use_gpu else -1  # Use GPU if available
        self.models = {}
        self.tokenizers = {}
        
        # Initialize models
        logger.info(f"Loading summarization model: {summarization_model}")
        self.summarizer = pipeline("summarization", 
                                  model=summarization_model, 
                                  device=self.device)
        
        logger.info(f"Loading relevance model: {relevance_model}")
        self.relevance_pipeline = pipeline("text-classification", 
                                          model=relevance_model,
                                          device=self.device)
        
        # Load QA model for answer assessment
        logger.info(f"Loading QA model: {qa_model}")
        self.qa_pipeline = pipeline("question-answering",
                                   model=qa_model,
                                   device=self.device)
        
    def summarize_text(self, text, query=None, max_length=150, min_length=50):
        """
        Summarize text using the loaded summarization model.
        
        Args:
            text (str): Text to summarize
            query (str, optional): Query to focus the summary
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            
        Returns:
            str: Generated summary
        """
        if not text or len(text) < 100:  # Don't summarize very short texts
            return text
            
        # If query is provided, prepend it to guide the summarization
        input_text = text
        if query:
            input_text = f"Query: {query}\n\nContent: {text}"
            
        # Truncate input if too long
        max_input_length = 1024  # Most models have context limits
        if len(input_text) > max_input_length:
            # Try to truncate at sentence boundaries
            sentences = sent_tokenize(input_text)
            truncated_text = ""
            for sentence in sentences:
                if len(truncated_text) + len(sentence) <= max_input_length:
                    truncated_text += sentence + " "
                else:
                    break
            input_text = truncated_text
        
        # Dynamically adjust max_length based on input length
        # For summarization, a common practice is to set max_length to about half the input length
        input_length = len(input_text.split())
        adjusted_max_length = min(max_length, max(min_length, input_length // 2))
        adjusted_min_length = min(min_length, max(1, adjusted_max_length // 2))
            
        try:
            summary = self.summarizer(input_text, 
                                     max_length=adjusted_max_length, 
                                     min_length=adjusted_min_length, 
                                     do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Fallback to the first few sentences
            sentences = sent_tokenize(text)
            return " ".join(sentences[:3])
    
    def assess_link_relevance(self, link_text, link_url, current_content, query):
        """
        Assess the relevance of a link to decide if it should be followed.
        
        Args:
            link_text (str): Text of the link
            link_url (str): URL of the link
            current_content (str): Content of the current page
            query (str): Search query
            
        Returns:
            float: Relevance score between 0 and 1
        """
        if not query:
            # Without a query, follow internal links with meaningful text
            return 0.7 if len(link_text) > 5 else 0.3
            
        # For relevance assessment, combine link text with URL components
        url_parts = urlparse(link_url).path.replace('-', ' ').replace('_', ' ').replace('/', ' ')
        assessment_text = f"{link_text} {url_parts}"
        
        try:
            # Use the relevance model to score this link against the query
            result = self.relevance_pipeline(
                [assessment_text, query],
                function_to_apply="sigmoid"
            )
            score = result[0]['score']
            return score
        except Exception as e:
            logger.error(f"Link relevance assessment error: {e}")
            # Fallback to basic text matching
            matches = sum(1 for term in query.lower().split() if term in assessment_text.lower())
            return min(0.9, matches * 0.2)  # Scale the score
            
    def assess_answer_completeness(self, query, content, confidence_threshold=0.8):
        """
        Assess whether the gathered content contains a complete answer to the query.
        
        Args:
            query (str): The search query or question
            content (str): The accumulated content to assess
            confidence_threshold (float): Threshold for confidence score
            
        Returns:
            tuple: (is_complete, confidence_score, answer)
        """
        if not query or not content:
            return False, 0.0, ""
            
        # Make sure the query is formatted as a question
        if not query.endswith('?'):
            # Try to reformulate as a question if it's not already one
            question_words = ['what', 'who', 'where', 'when', 'why', 'how']
            if not any(query.lower().startswith(word) for word in question_words):
                # Prepend "What is" or similar if not a question
                if query.lower().startswith('is') or query.lower().startswith('are'):
                    question = query + '?'
                else:
                    question = f"What is {query}?"
            else:
                question = query + '?'
        else:
            question = query
            
        try:
            # Use QA model to find answer in content
            result = self.qa_pipeline(
                question=question,
                context=content[:10000],  # Limit context length
                handle_impossible_answer=True
            )
            
            answer = result['answer']
            confidence = result['score']
            
            # Determine if answer is complete based on confidence and length
            is_complete = (confidence >= confidence_threshold and len(answer) >= 10)
            
            return is_complete, confidence, answer
            
        except Exception as e:
            logger.error(f"Answer assessment error: {e}")
            return False, 0.0, ""
    
    def extract_key_information(self, page_content, query):
        """
        Extract key information from page content that's relevant to the query.
        
        Args:
            page_content (str): Full page content
            query (str): Search query
            
        Returns:
            dict: Dictionary with key information
        """
        # First, generate a focused summary
        summary = self.summarize_text(page_content, query)
        
        # Extract key sentences using the summary as a guide
        sentences = sent_tokenize(page_content)
        key_sentences = []
        
        # Score sentences based on similarity to summary and query terms
        summary_terms = set(word.lower() for word in summary.split() 
                           if len(word) > 3 and word.lower() not in stopwords.words('english'))
        query_terms = set(word.lower() for word in query.split() 
                         if len(word) > 3 and word.lower() not in stopwords.words('english'))
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            # Calculate relevance scores
            summary_score = sum(1 for term in summary_terms if term in sentence.lower()) / max(1, len(summary_terms))
            query_score = sum(1 for term in query_terms if term in sentence.lower()) / max(1, len(query_terms))
            
            # Combined score
            score = 0.6 * summary_score + 0.4 * query_score
            
            if score > 0.2:  # Only include somewhat relevant sentences
                key_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        key_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sentence for sentence, _ in key_sentences[:7]]
        
        return {
            'summary': summary,
            'key_sentences': top_sentences
        }


class CrawlResult:
    """Class representing the result of crawling a website."""
    
    def __init__(self, url, title=None, content=None, metadata=None, important_sections=None,
                key_sentences=None, links=None, resource_type="webpage", summary=None):
        """Initialize a crawl result."""
        self.url = url
        self.title = title or ""
        self.content = content or ""
        self.metadata = metadata or {}
        self.important_sections = important_sections or []
        self.key_sentences = key_sentences or []
        self.links = links or []
        self.resource_type = resource_type
        self.importance_score = 0
        self.crawl_time = time.time()
        self.summary = summary or ""
        
        # Answer-related attributes
        self.is_answer = False
        self.answer = ""
        self.answer_confidence = 0.0
        self.partial_answer = ""
        self.early_stopping_triggered = False
        
    def to_dict(self):
        """Convert to dictionary representation."""
        result = {
            'url': self.url,
            'title': self.title,
            'content_summary': self.content[:300] + "..." if len(self.content) > 300 else self.content,
            'metadata': self.metadata,
            'important_sections': self.important_sections,
            'key_sentences': self.key_sentences[:5],
            'related_links': self.links[:10],
            'resource_type': self.resource_type,
            'importance_score': self.importance_score,
            'crawl_time': self.crawl_time,
            'summary': self.summary
        }
        
        # Add answer information if available
        if self.is_answer:
            result['is_answer'] = True
            result['answer'] = self.answer
            result['answer_confidence'] = self.answer_confidence
            result['early_stopping_triggered'] = self.early_stopping_triggered
        elif self.partial_answer:
            result['partial_answer'] = self.partial_answer
            result['answer_confidence'] = self.answer_confidence
            
        return result
        
    def __str__(self):
        """String representation of the crawl result."""
        base_str = f"CrawlResult(url={self.url}, title={self.title}, type={self.resource_type}, score={self.importance_score}"
        
        if self.is_answer:
            base_str += f", ANSWER(confidence={self.answer_confidence:.2f})"
        elif self.partial_answer:
            base_str += f", PARTIAL_ANSWER(confidence={self.answer_confidence:.2f})"
            
        return base_str + ")"


class RateLimiter:
    """Rate limiter for ethical web crawling."""
    
    def __init__(self, requests_per_second=1, respect_robots=True):
        """Initialize the rate limiter."""
        self.requests_per_second = requests_per_second
        self.respect_robots = respect_robots
        self.last_request_time = defaultdict(float)
        self.robot_parsers = {}
        self.domain_rates = {}
        
    def wait(self, url):
        """Wait appropriate time before making a request to the URL."""
        domain = urlparse(url).netloc
        
        # Get delay for this domain
        delay = 1.0 / self.requests_per_second
        
        # Check if we have a custom rate for this domain
        if domain in self.domain_rates:
            delay = 1.0 / self.domain_rates[domain]
        
        # Check robots.txt if enabled
        if self.respect_robots:
            robots_delay = self.get_robots_delay(url)
            if robots_delay is not None:
                delay = max(delay, robots_delay)
        
        # Calculate wait time
        current_time = time.time()
        elapsed = current_time - self.last_request_time[domain]
        wait_time = max(0, delay - elapsed)
        
        # Add small random delay to avoid patterns
        wait_time += random.uniform(0, 0.5)
        
        # Wait if needed
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time[domain] = time.time()
    
    def get_robots_delay(self, url):
        """Get crawl delay from robots.txt file."""
        domain = urlparse(url).netloc
        
        # Return cached parser if available
        if domain in self.robot_parsers:
            return self.get_delay_from_parser(self.robot_parsers[domain])
        
        # Create new parser
        robots_url = f"https://{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        
        try:
            parser.read()
            self.robot_parsers[domain] = parser
            return self.get_delay_from_parser(parser)
        except Exception as e:
            logger.warning(f"Error reading robots.txt from {domain}: {e}")
            return None
    
    def get_delay_from_parser(self, parser):
        """Extract delay from robot parser."""
        # Try with different user agents
        for agent in ['*', 'DeepResearch', 'DeepResearchBot']:
            delay = parser.crawl_delay(agent)
            if delay:
                return delay
        return None
    
    def can_fetch(self, url):
        """Check if the URL can be fetched according to robots.txt."""
        if not self.respect_robots:
            return True
        
        domain = urlparse(url).netloc
        if domain not in self.robot_parsers:
            self.get_robots_delay(url)
        
        if domain in self.robot_parsers:
            return self.robot_parsers[domain].can_fetch('DeepResearch', url)
        
        return True
    
    def set_domain_rate(self, domain, requests_per_second):
        """Set custom rate limit for a specific domain."""
        self.domain_rates[domain] = requests_per_second


class Fetcher:
    """Fetches content from URLs."""
    
    def __init__(self, rate_limiter=None, timeout=10, cache_dir=None, max_retries=2):
        """Initialize the fetcher."""
        self.rate_limiter = rate_limiter or RateLimiter()
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set up caching if enabled
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DeepResearchBot/1.0; +https://example.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })
    
    def fetch(self, url, use_cache=True, force_refetch=False):
        """
        Fetch content from a URL.
        
        Args:
            url (str): URL to fetch
            use_cache (bool): Whether to use cache
            force_refetch (bool): Whether to force a refetch
            
        Returns:
            tuple: (content, headers, status_code)
        """
        # Check cache if enabled
        if self.cache_dir and use_cache and not force_refetch:
            cached_content = self._get_from_cache(url)
            if cached_content:
                return cached_content
        
        # Check if we're allowed to fetch this URL
        if not self.rate_limiter.can_fetch(url):
            logger.warning(f"Not allowed to fetch {url} according to robots.txt")
            return None, None, 403
        
        # Wait according to rate limiting
        self.rate_limiter.wait(url)
        
        # Attempt to fetch the URL
        content, headers, status_code = None, None, None
        retries = 0
        
        while retries <= self.max_retries:
            try:
                response = self.session.get(url, timeout=self.timeout)
                content = response.content
                headers = dict(response.headers)
                status_code = response.status_code
                
                # If successful, break the retry loop
                if 200 <= status_code < 300:
                    break
                
                # Special handling for redirects
                if 300 <= status_code < 400 and 'Location' in headers:
                    redirect_url = urljoin(url, headers['Location'])
                    logger.info(f"Following redirect from {url} to {redirect_url}")
                    url = redirect_url
                    retries += 1
                    continue
                
                # For other non-success status codes
                logger.warning(f"Received status code {status_code} for {url}")
                break
                
            except requests.RequestException as e:
                logger.warning(f"Error fetching {url}: {e}")
                retries += 1
                # Add exponential backoff
                if retries <= self.max_retries:
                    time.sleep(2 ** retries)
        
        # Cache the result if caching is enabled
        if self.cache_dir and content and status_code == 200:
            self._save_to_cache(url, content, headers, status_code)
        
        return content, headers, status_code
    
    def _get_cache_path(self, url):
        """Get the cache file path for a URL."""
        if not self.cache_dir:
            return None
        
        # Create a hash of the URL to use as filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.cache")
    
    def _get_from_cache(self, url):
        """Get cached content for a URL."""
        cache_path = self._get_cache_path(url)
        if not cache_path or not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache has expired (older than 24 hours)
            if time.time() - cache_data['timestamp'] > 86400:
                return None
            
            return cache_data['content'].encode('utf-8'), cache_data['headers'], cache_data['status_code']
        except Exception as e:
            logger.warning(f"Error reading cache for {url}: {e}")
            return None
    
    def _save_to_cache(self, url, content, headers, status_code):
        """Save content to cache."""
        if not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(url)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'url': url,
                    'content': content.decode('utf-8', errors='replace'),
                    'headers': headers,
                    'status_code': status_code,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Error saving cache for {url}: {e}")


class ContentExtractor:
    """Extracts content from fetched web pages."""
    
    def __init__(self, model_handler=None):
        """Initialize the content extractor."""
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.model_handler = model_handler
    
    def extract(self, content, url, content_type='text/html', query=None):
        """
        Extract content from raw fetched content.
        
        Args:
            content (bytes): Raw content
            url (str): Source URL
            content_type (str): Content type
            query (str): Optional query to focus extraction
            
        Returns:
            dict: Extracted content
        """
        # Check content type and use appropriate extractor
        if content_type.startswith('text/html'):
            extracted = self._extract_html(content, url)
        elif content_type.startswith('application/pdf'):
            extracted = self._extract_pdf(content, url)
        elif content_type.startswith('text/plain'):
            extracted = self._extract_text(content, url)
        elif 'json' in content_type:
            extracted = self._extract_json(content, url)
        else:
            logger.warning(f"Unsupported content type: {content_type}")
            extracted = {
                'title': url,
                'text': "",
                'links': [],
                'metadata': {'content_type': content_type}
            }
        
        # Use model to extract key sentences if available
        if self.model_handler and query and extracted['text']:
            try:
                key_info = self.model_handler.extract_key_information(extracted['text'], query)
                extracted['key_sentences'] = key_info['key_sentences']
                extracted['summary'] = key_info['summary']
            except Exception as e:
                logger.error(f"Error extracting key information: {e}")
                # Fallback to simple key sentence extraction
                extracted['key_sentences'] = self.extract_key_sentences(extracted['text'], query)
                extracted['summary'] = " ".join(extracted['key_sentences'][:3])
        else:
            # Fallback to simple key sentence extraction
            extracted['key_sentences'] = self.extract_key_sentences(extracted['text'], query)
            extracted['summary'] = " ".join(extracted['key_sentences'][:3])
            
        return extracted
    
    def _extract_html(self, content, url):
        """Extract content from HTML."""
        try:
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'iframe', 'noscript']):
                element.decompose()
            
            # Get title
            title = soup.title.string if soup.title else ""
            
            # Get metadata
            metadata = self._extract_html_metadata(soup)
            
            # Get main content
            main_content = self._extract_main_content(soup)
            
            # Get links
            links = self._extract_links(soup, url)
            
            # Get important sections
            sections = self._extract_sections(soup)
            
            return {
                'title': title,
                'text': main_content,
                'links': links,
                'metadata': metadata,
                'sections': sections
            }
            
        except Exception as e:
            logger.warning(f"Error extracting HTML content from {url}: {e}")
            return {
                'title': url,
                'text': "",
                'links': [],
                'metadata': {}
            }
    
    def _extract_html_metadata(self, soup):
        """Extract metadata from HTML."""
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        # Extract publication date
        pub_date = None
        date_patterns = [
            soup.select_one('time[datetime]'),
            soup.select_one('meta[property="article:published_time"]'),
            soup.select_one('span[itemprop="datePublished"]'),
            soup.select_one('.date'),
            soup.select_one('.published'),
            soup.select_one('.post-date')
        ]
        
        for pattern in date_patterns:
            if pattern:
                if pattern.name == 'time' and pattern.get('datetime'):
                    pub_date = pattern.get('datetime')
                    break
                elif pattern.name == 'meta' and pattern.get('content'):
                    pub_date = pattern.get('content')
                    break
                else:
                    pub_date = pattern.text.strip()
                    break
        
        if pub_date:
            metadata['published_date'] = pub_date
        
        # Extract author
        author = None
        author_patterns = [
            soup.select_one('meta[name="author"]'),
            soup.select_one('meta[property="article:author"]'),
            soup.select_one('span[itemprop="author"]'),
            soup.select_one('.author'),
            soup.select_one('.byline')
        ]
        
        for pattern in author_patterns:
            if pattern:
                if pattern.name == 'meta' and pattern.get('content'):
                    author = pattern.get('content')
                    break
                else:
                    author = pattern.text.strip()
                    break
        
        if author:
            metadata['author'] = author
        
        return metadata
    
    def _extract_main_content(self, soup):
        """Extract main content from HTML."""
        # Try to find the main content area using common selectors
        main_content_selectors = [
            'article', 'main', '.content', '.post-content', '.entry-content',
            '#content', '.article', '.post', '.entry', '.main-content'
        ]
        
        for selector in main_content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                return content_element.get_text(separator=' ', strip=True)
        
        # If no main content area found, use the body text
        return soup.body.get_text(separator=' ', strip=True) if soup.body else ""
    
    def _extract_links(self, soup, url):
        """Extract links from HTML."""
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True)
            
            # Create absolute URL
            abs_url = urljoin(url, href)
            
            # Skip mailto:, javascript:, etc.
            if abs_url.startswith(('http://', 'https://')):
                links.append({
                    'url': abs_url,
                    'text': text,
                    'internal': urlparse(abs_url).netloc == urlparse(url).netloc
                })
        
        return links
    
    def _extract_sections(self, soup):
        """Extract sections from HTML."""
        sections = []
        
        # Find headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = heading.get_text(strip=True)
            
            # Skip empty headings
            if not heading_text:
                continue
            
            # Get the content following this heading
            content = []
            for sibling in heading.next_siblings:
                if sibling.name in ['h1', 'h2', 'h3']:
                    break
                if sibling.name and sibling.get_text(strip=True):
                    content.append(sibling.get_text(strip=True))
            
            sections.append({
                'heading': heading_text,
                'level': int(heading.name[1]),
                'content': ' '.join(content)
            })
        
        return sections
    
    def _extract_pdf(self, content, url):
        """Extract content from PDF."""
        try:
            # This requires an external library like PyPDF2 or pdfminer
            # For simplicity, we'll just return a placeholder
            return {
                'title': f"PDF: {os.path.basename(url)}",
                'text': "PDF content extraction requires additional libraries",
                'links': [],
                'metadata': {'content_type': 'application/pdf'}
            }
        except Exception as e:
            logger.warning(f"Error extracting PDF content from {url}: {e}")
            return {
                'title': url,
                'text': "",
                'links': [],
                'metadata': {'content_type': 'application/pdf'}
            }
    
    def _extract_text(self, content, url):
        """Extract content from plain text."""
        try:
            text = content.decode('utf-8', errors='replace')
            lines = text.split('\n')
            
            # Try to extract a title from the first line
            title = lines[0].strip() if lines else os.path.basename(url)
            
            return {
                'title': title,
                'text': text,
                'links': [],
                'metadata': {'content_type': 'text/plain'}
            }
        except Exception as e:
            logger.warning(f"Error extracting text content from {url}: {e}")
            return {
                'title': url,
                'text': "",
                'links': [],
                'metadata': {'content_type': 'text/plain'}
            }
    
    def _extract_json(self, content, url):
        """Extract content from JSON."""
        try:
            data = json.loads(content)
            
            # Try to extract a title
            title = data.get('title', os.path.basename(url))
            
            # Try to extract text content
            text = ""
            if isinstance(data, dict):
                # Look for common text fields
                for field in ['content', 'text', 'body', 'description']:
                    if field in data and isinstance(data[field], str):
                        text = data[field]
                        break
            
            return {
                'title': title,
                'text': text if text else json.dumps(data, indent=2),
                'links': [],
                'metadata': {'content_type': 'application/json'}
            }
        except Exception as e:
            logger.warning(f"Error extracting JSON content from {url}: {e}")
            return {
                'title': url,
                'text': "",
                'links': [],
                'metadata': {'content_type': 'application/json'}
            }
    
    def extract_key_sentences(self, text, query=None, num_sentences=5):
        """Extract key sentences from text, optionally focusing on query relevance."""
        if not text:
            return []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Score sentences
        scores = {}
        
        for i, sentence in enumerate(sentences):
            
            # Initial position-based score (earlier sentences often more important)
            position_score = 1.0 / (i + 1)
            
            # Word frequency score
            words = word_tokenize(sentence.lower())
            # Avoid division by zero
            word_count = max(1, len(words))  # Ensure at least 1
            word_freq_score = sum(1 for word in words if word.isalnum() and word not in self.stop_words) / word_count
            
            # Query relevance score
            query_score = 0
            if query:
                query_words = set(word.lower() for word in query.split() if word.lower() not in self.stop_words)
                # Avoid division by zero
                if query_words:  # Only calculate if we have query words
                    query_score = sum(1 for word in words if word in query_words) / len(query_words)
            
            # Combined score
            scores[i] = 0.2 * position_score + 0.3 * word_freq_score + 0.5 * query_score if query else 0.4 * position_score + 0.6 * word_freq_score
        
        # Get top sentences
        top_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        top_indices.sort()  # Keep original order
        
        return [sentences[i] for i in top_indices]


class Navigator:
    """Navigates through websites by following links."""
    
    def __init__(self, fetcher, extractor, model_handler=None, max_depth=2, max_pages_per_site=10):
        """Initialize the navigator."""
        self.fetcher = fetcher
        self.extractor = extractor
        self.max_depth = max_depth
        self.max_pages_per_site = max_pages_per_site
        self.model_handler = model_handler
    
    def crawl(self, url, query=None, follow_links=True, early_stopping=True):
        """
        Crawl a website starting from the given URL.
        
        Args:
            url (str): Starting URL
            query (str): Optional query to focus the crawl
            follow_links (bool): Whether to follow links
            early_stopping (bool): Whether to stop when a complete answer is found
            
        Returns:
            list: List of CrawlResult objects
        """
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL scheme: {url}")
            return []
        
        # Get the domain
        domain = urlparse(url).netloc
        
        # Initialize crawl state
        visited = set()
        to_visit = [(url, 0)]  # (url, depth)
        results = []
        site_page_count = 0
        
        # Variables for early stopping
        answer_found = False
        cumulative_content = ""
        best_answer = ""
        best_confidence = 0.0
        
        # Process URLs in queue
        while to_visit and site_page_count < self.max_pages_per_site and not answer_found:
            current_url, depth = to_visit.pop(0)
            
            # Skip if already visited
            if current_url in visited:
                continue
            
            # Skip if not allowed by robots.txt
            if not self.fetcher.rate_limiter.can_fetch(current_url):
                continue
            
            # Mark as visited
            visited.add(current_url)
            
            # Fetch content
            logger.info(f"Crawling {current_url} (depth {depth})")
            content, headers, status_code = self.fetcher.fetch(current_url)
            
            # Skip if fetch failed
            if not content or status_code != 200:
                continue
            
            # Get content type
            content_type = headers.get('Content-Type', 'text/html')
            
            # Extract content
            extracted = self.extractor.extract(content, current_url, content_type, query)
            
            # Create crawl result
            result = CrawlResult(
                url=current_url,
                title=extracted['title'],
                content=extracted['text'],
                metadata=extracted['metadata'],
                important_sections=extracted.get('sections', []),
                key_sentences=extracted.get('key_sentences', []),
                links=extracted['links'],
                resource_type=self._determine_resource_type(content_type),
                summary=extracted.get('summary', '')
            )
            
            # Add relevance score based on query
            if query:
                result.importance_score = self._calculate_relevance(result, query)
            
            # Add to results
            results.append(result)
            site_page_count += 1
            
            # Check for early stopping if enabled
            if early_stopping and query and self.model_handler:
                # Update cumulative content with the most relevant information
                # We focus on key sentences and summaries to avoid noise
                if result.summary:
                    cumulative_content += f"\n\n{result.summary}"
                elif result.key_sentences:
                    cumulative_content += f"\n\n{' '.join(result.key_sentences)}"
                else:
                    # If no summary or key sentences, use a sample of the content
                    content_sample = result.content[:2000]  # Use first 2000 chars to avoid too much noise
                    cumulative_content += f"\n\n{content_sample}"
                
                # Check if we have enough information to answer the query
                is_complete, confidence, answer = self.model_handler.assess_answer_completeness(
                    query, cumulative_content
                )
                
                # Update best answer if this one is better
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_answer = answer
                
                # If we have a complete answer, stop crawling
                if is_complete:
                    logger.info(f"Complete answer found with confidence {confidence:.2f}: {answer}")
                    result.is_answer = True
                    result.answer_confidence = confidence
                    result.answer = answer
                    answer_found = True
                    break
            
            # Follow links if enabled, no answer found yet, and not too deep
            if follow_links and depth < self.max_depth and not answer_found:
                links_to_follow = self._prioritize_links(extracted['links'], query, current_url, extracted['text'])
                
                for link, score in links_to_follow:
                    link_url = link['url']
                    link_domain = urlparse(link_url).netloc
                    
                    # Only follow links within same domain
                    if link_domain == domain and link_url not in visited:
                        to_visit.append((link_url, depth + 1))
        
        # Add answer information to results if found
        if answer_found and results:
            # Mark the last result with the answer
            for r in results:
                if hasattr(r, 'is_answer') and r.is_answer:
                    r.early_stopping_triggered = True
        elif best_answer and best_confidence > 0.5 and results:
            # If we didn't trigger early stopping but still found a decent answer
            results[0].partial_answer = best_answer
            results[0].answer_confidence = best_confidence
        
        # Sort results by relevance
        if query:
            results.sort(key=lambda r: r.importance_score, reverse=True)
        
        return results
    
    def _determine_resource_type(self, content_type):
        """Determine resource type from content type."""
        if 'html' in content_type:
            return 'webpage'
        elif 'pdf' in content_type:
            return 'pdf'
        elif 'json' in content_type:
            return 'json'
        elif 'text' in content_type:
            return 'text'
        elif 'image' in content_type:
            return 'image'
        else:
            return 'other'
    
    def _calculate_relevance(self, result, query):
        """Calculate relevance score of a result to the query."""
        # Use model-based relevance if available
        if self.model_handler:
            try:
                # Create a sample from the content for relevance assessment
                content_sample = result.title + " " + result.content[:1000]
                
                # Let the model score this content against the query
                score = self.model_handler.assess_link_relevance(
                    result.title,
                    result.url, 
                    content_sample,
                    query
                )
                return score
            except Exception as e:
                logger.error(f"Model-based relevance calculation error: {e}")
                # Fall back to traditional relevance calculation
        
        # Traditional TF-based relevance
        query_terms = set(term.lower() for term in query.split() if term.lower() not in self.extractor.stop_words)
        
        if not query_terms:
            return 0
        
        # Check title
        title_score = sum(1 for term in query_terms if term in result.title.lower()) / len(query_terms)
        
        # Check content (use a sample to speed up processing)
        content_sample = result.content[:5000].lower()
        content_words = content_sample.split()
        # Avoid division by zero
        word_count = max(1, len(content_words))  # Ensure at least 1 word
        content_score = sum(content_sample.count(term) for term in query_terms) / (word_count / 100)
        
        # Check key sentences
        sentence_score = sum(1 for sentence in result.key_sentences for term in query_terms if term in sentence.lower()) / (len(result.key_sentences) or 1)
        
        # Combine scores
        return 0.3 * title_score + 0.5 * content_score + 0.2 * sentence_score
    
    def _prioritize_links(self, links, query, current_url, page_content):
        """Prioritize links for crawling based on relevance to query using AI model."""
        if not query:
            # Without a query, return internal links first
            return [(link, 0.5) for link in links if link.get('internal', False)]
        
        scored_links = []
        
        # Use model for link prioritization if available
        if self.model_handler:
            for link in links:
                # Skip non-internal links
                if not link.get('internal', False):
                    continue
                
                # Use model to assess link relevance
                try:
                    score = self.model_handler.assess_link_relevance(
                        link.get('text', ''), 
                        link['url'],
                        page_content,
                        query
                    )
                    scored_links.append((link, score))
                except Exception as e:
                    logger.error(f"Link relevance assessment error: {e}")
                    # Fallback to simple matching
                    query_terms = set(term.lower() for term in query.split() if term not in self.extractor.stop_words)
                    link_text = link.get('text', '').lower()
                    score = sum(1 for term in query_terms if term in link_text) / max(1, len(query_terms))
                    scored_links.append((link, score))
        else:
            # Traditional prioritization
            query_terms = set(term.lower() for term in query.split() if term not in self.extractor.stop_words)
            
            for link in links:
                # Skip non-internal links
                if not link.get('internal', False):
                    continue
                
                # Calculate relevance to query
                link_text = link.get('text', '').lower()
                score = sum(1 for term in query_terms if term in link_text) / max(1, len(query_terms))
                
                scored_links.append((link, score))
        
        # Sort by score
        scored_links.sort(key=lambda x: x[1], reverse=True)
        
        # Return top links (with a minimum score threshold)
        return [(link, score) for link, score in scored_links if score > 0.2]


class WebCrawler:
    """Main crawler class that combines fetching, extracting, and navigating."""
    
    def __init__(self, cache_dir=None, respect_robots=True, requests_per_second=1, 
                 max_depth=2, max_pages_per_site=10,
                 summarization_model="facebook/bart-large-cnn",
                 relevance_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_gpu=True):
        """Initialize the web crawler with AI models."""
        # Initialize models
        self.model_handler = ModelHandler(
            summarization_model=summarization_model,
            relevance_model=relevance_model,
            use_gpu=use_gpu
        )
        
        # Initialize components
        self.rate_limiter = RateLimiter(requests_per_second=requests_per_second, respect_robots=respect_robots)
        self.fetcher = Fetcher(rate_limiter=self.rate_limiter, cache_dir=cache_dir)
        self.extractor = ContentExtractor(model_handler=self.model_handler)
        self.navigator = Navigator(
            self.fetcher, 
            self.extractor, 
            model_handler=self.model_handler,
            max_depth=max_depth, 
            max_pages_per_site=max_pages_per_site
        )
    
    def crawl_url(self, url, query=None, follow_links=True, early_stopping=True):
        """
        Crawl a single URL and its links.
        
        Args:
            url (str): URL to crawl
            query (str): Optional query to focus crawling
            follow_links (bool): Whether to follow links
            early_stopping (bool): Whether to stop when a complete answer is found
            
        Returns:
            list: CrawlResult objects
        """
        return self.navigator.crawl(url, query, follow_links, early_stopping)
    
    def crawl_search_results(self, search_results, query=None, max_sites=5, follow_links=True, early_stopping=True):
        """
        Crawl websites from search results.
        
        Args:
            search_results (list): List of search result objects
            query (str): Optional query to focus crawling
            max_sites (int): Maximum number of sites to crawl
            follow_links (bool): Whether to follow links
            early_stopping (bool): Whether to stop when a complete answer is found
            
        Returns:
            list: CrawlResult objects
        """
        results = []
        sites_crawled = 0
        answer_found = False
        
        # Process each search result
        for result in search_results:
            # Skip if max sites reached or answer found
            if sites_crawled >= max_sites or answer_found:
                break
            
            # Get URL from search result
            url = result.url if hasattr(result, 'url') else result.get('url', None)
            if not url:
                continue
            
            # Skip non-webpage results
            content_type = result.content_type if hasattr(result, 'content_type') else result.get('content_type', 'webpage')
            if content_type != 'webpage':
                continue
            
            # Crawl the site
            site_results = self.crawl_url(url, query, follow_links, early_stopping)
            if site_results:
                results.extend(site_results)
                sites_crawled += 1
                
                # Check if an answer was found
                if early_stopping:
                    for r in site_results:
                        if hasattr(r, 'is_answer') and r.is_answer:
                            answer_found = True
                            break
                
                # If answer found, no need to crawl more sites
                if answer_found:
                    logger.info("Answer found, stopping search result crawling")
                    break
        
        return results
    
    def synthesize_information(self, crawl_results, search_results=None, query=None):
        """
        Synthesize information from crawl results using Hugging Face models.
        
        Args:
            crawl_results (list): List of CrawlResult objects
            search_results (list): Original search results with snippets
            query (str): Original search query
            
        Returns:
            dict: Synthesized information
        """
        if not crawl_results and not search_results:
            return {
                'success': False,
                'message': 'No information to synthesize'
            }
            
        # Check if we have a definitive answer from early stopping
        for result in crawl_results:
            if hasattr(result, 'is_answer') and result.is_answer:
                return {
                    'success': True,
                    'query': query,
                    'answer_found': True,
                    'answer': result.answer,
                    'confidence': result.answer_confidence,
                    'source': {
                        'url': result.url,
                        'title': result.title
                    },
                    'early_stopping': result.early_stopping_triggered,
                    'additional_sources': [
                        {'url': r.url, 'title': r.title}
                        for r in crawl_results if r != result
                    ][:5]
                }
                
        # Check for a partial answer
        partial_answers = [r for r in crawl_results if hasattr(r, 'partial_answer') and r.partial_answer]
        has_partial_answer = False
        
        if partial_answers:
            best_partial = max(partial_answers, key=lambda r: r.answer_confidence)
            has_partial_answer = True
            # Still do full synthesis, but include the partial answer
        
        # Collect all content
        all_content = []
        
        # Add content from search results snippets if available
        if search_results:
            snippets = []
            for result in search_results:
                snippet = None
                title = None
                
                # Handle different result object structures
                if hasattr(result, 'snippet'):
                    snippet = result.snippet
                    title = result.title
                elif isinstance(result, dict):
                    snippet = result.get('snippet', '')
                    title = result.get('title', '')
                
                if snippet:
                    snippets.append(f"{title}: {snippet}")
            
            if snippets:
                all_content.append("Search Result Snippets:\n" + "\n".join(snippets))
        
        # Add content from crawl results
        for result in crawl_results:
            # Use the summary if available, otherwise key sentences
            if hasattr(result, 'summary') and result.summary:
                all_content.append(f"From {result.title} ({result.url}):\n{result.summary}")
            elif hasattr(result, 'key_sentences') and result.key_sentences:
                all_content.append(f"From {result.title} ({result.url}):\n" + "\n".join(result.key_sentences))
        
        # Combine all content with reasonable limits
        combined_text = "\n\n".join(all_content)
        
        # Use the model to generate a comprehensive summary
        max_combined_length = 10000  # Most models have context limits
        if len(combined_text) > max_combined_length:
            # Truncate at paragraph boundaries
            paragraphs = combined_text.split("\n\n")
            truncated_text = ""
            for para in paragraphs:
                if len(truncated_text) + len(para) + 2 <= max_combined_length:
                    truncated_text += para + "\n\n"
                else:
                    break
            combined_text = truncated_text
        
        try:
            # Generate comprehensive synthesis
            synthesis_prompt = f"Query: {query}\n\nSynthesize the following information into a comprehensive analysis:\n\n{combined_text}"
            synthesis = self.model_handler.summarize_text(
                synthesis_prompt,
                max_length=300,  # Reduced max_length to avoid warnings
                min_length=100   # Reduced min_length to be proportional
            )
            
            # Generate key points
            key_points_prompt = f"Based on this information, what are the 5 most important points about: {query}\n\n{combined_text}"
            key_points_text = self.model_handler.summarize_text(
                key_points_prompt,
                max_length=200,  # Reduced max_length to avoid warnings
                min_length=50    # Reduced min_length to be proportional
            )
            
            # Extract key points as a list
            key_points = [point.strip() for point in key_points_text.split('\n') if point.strip()]
            if not key_points:
                # If splitting by newlines didn't work, try to split by numbers or bullet points
                key_points = re.findall(r'(?:^|\n)(?:\d+\.\s*|\*\s*)(.*?)(?=(?:\n\d+\.|\n\*|$))', key_points_text)
            if not key_points:
                # Last resort: just split by sentences and take top 5
                key_points = sent_tokenize(key_points_text)[:5]
            
            # Collect sources
            sources = []
            for result in crawl_results:
                if hasattr(result, 'title') and hasattr(result, 'url'):
                    sources.append({
                        'title': result.title,
                        'url': result.url,
                        'relevance': getattr(result, 'importance_score', 0)
                    })
            
            # Sort sources by relevance
            sources.sort(key=lambda x: x.get('relevance', 0), reverse=True)
            
            result = {
                'success': True,
                'query': query,
                'synthesis': synthesis,
                'key_points': key_points,
                'sources': sources[:10],  # Limit to top 10 sources
                'crawled_pages': len(crawl_results),
                'model_info': {
                    'summarization': self.model_handler.summarizer.model.config.model_type,
                    'relevance': self.model_handler.relevance_pipeline.model.config.model_type,
                    'qa': self.model_handler.qa_pipeline.model.config.model_type
                }
            }
            
            # Include partial answer if available
            if has_partial_answer:
                result['partial_answer'] = {
                    'text': best_partial.partial_answer,
                    'confidence': best_partial.answer_confidence,
                    'source': {
                        'url': best_partial.url,
                        'title': best_partial.title
                    }
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            
            # Fallback: simple extraction of key sentences from all results
            all_sentences = []
            for result in crawl_results:
                if hasattr(result, 'key_sentences'):
                    for sentence in result.key_sentences:
                        all_sentences.append((sentence, result.url, getattr(result, 'importance_score', 0)))
            
            # Sort by result importance score
            all_sentences.sort(key=lambda x: x[2], reverse=True)
            
            # Take top sentences
            top_sentences = all_sentences[:10]
            
            # Get sources
            sources = []
            for result in crawl_results:
                if hasattr(result, 'title') and hasattr(result, 'url'):
                    sources.append({
                        'title': result.title,
                        'url': result.url,
                        'relevance': getattr(result, 'importance_score', 0)
                    })
            
            # Sort sources by relevance
            sources.sort(key=lambda x: x.get('relevance', 0), reverse=True)
            
            return {
                'success': True,
                'query': query,
                'synthesis': "Error generating synthesis with model. Here are key extracted sentences.",
                'key_points': [sentence for sentence, _, _ in top_sentences],
                'sources': [(url, sentence) for sentence, url, _ in top_sentences],
                'all_sources': sources[:10],  # Limit to top 10 sources
                'fallback': True
            }