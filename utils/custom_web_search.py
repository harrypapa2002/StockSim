"""
Enhanced custom web search implementation that provides better error reporting
"""

import asyncio
import aiohttp
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import logging


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def _extract_article_content(soup: BeautifulSoup) -> str:
    """Extract the main article content from HTML"""

    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
        element.decompose()

    # Try different content extraction strategies
    content_selectors = [
        'article',
        '[role="main"]',
        '.article-content',
        '.content',
        '.post-content',
        '.entry-content',
        '.article-body',
        '.story-body',
        'main',
        '#content',
        '.main-content'
    ]

    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            # Get the largest element (most likely to be main content)
            largest_element = max(elements, key=lambda x: len(x.get_text()))
            content = largest_element.get_text()
            if len(content.strip()) > 200:  # Reasonable content length
                return _clean_text(content)

    # Fallback: get all paragraph text
    paragraphs = soup.find_all('p')
    if paragraphs:
        content = ' '.join([p.get_text() for p in paragraphs])
        if len(content.strip()) > 100:
            return _clean_text(content)

    # Last resort: get body text
    body = soup.find('body')
    if body:
        return _clean_text(body.get_text())

    return _clean_text(soup.get_text())


class CustomWebSearcher:
    """Custom web search implementation for extracting content from news URLs"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def fetch_url_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract content from a single URL

        Returns:
            Dict with keys: url, title, content, success, error
        """
        try:
            self.logger.info(f"Fetching content from: {url}")

            async with self.session.get(url) as response:
                if response.status != 200:
                    error_msg = f'HTTP {response.status}'
                    self.logger.warning(f"Failed to fetch {url}: {error_msg}")
                    return {
                        'url': url,
                        'title': '',
                        'content': '',
                        'success': False,
                        'error': error_msg
                    }

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else ''

                # Extract main content - try multiple strategies
                content = _extract_article_content(soup)

                # Check if we got meaningful content
                if len(content.strip()) < 50:
                    error_msg = f"Insufficient content extracted ({len(content)} chars)"
                    self.logger.warning(f"Failed to extract meaningful content from {url}: {error_msg}")
                    return {
                        'url': url,
                        'title': title,
                        'content': content,
                        'success': False,
                        'error': error_msg
                    }

                self.logger.info(f"Successfully extracted {len(content)} characters from {url}")

                return {
                    'url': url,
                    'title': title,
                    'content': content[:10000],
                    'success': True,
                    'error': None
                }

        except asyncio.TimeoutError:
            error_msg = "Timeout (30s exceeded)"
            self.logger.warning(f"Timeout fetching {url}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': error_msg
            }
        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error: {str(e)}"
            self.logger.error(f"HTTP error fetching {url}: {error_msg}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"Error fetching {url}: {error_msg}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': error_msg
            }

    async def search_urls(self, urls: List[str], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Search multiple URLs concurrently

        Args:
            urls: List of URLs to fetch
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of ALL results (both successful and failed)
        """
        if not urls:
            return []

        self.logger.info(f"Starting web search for {len(urls)} URLs with max_concurrent={max_concurrent}")

        # Limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url):
            async with semaphore:
                return await self.fetch_url_content(url)

        # Execute all requests concurrently
        results = await asyncio.gather(
            *[fetch_with_semaphore(url) for url in urls],
            return_exceptions=True
        )

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions that occurred during the gather
                error_msg = f"Exception during fetch: {str(result)}"
                self.logger.error(f"Exception fetching {urls[i]}: {error_msg}")
                processed_results.append({
                    'url': urls[i],
                    'title': '',
                    'content': '',
                    'success': False,
                    'error': error_msg
                })
            elif isinstance(result, dict):
                processed_results.append(result)
            else:
                # Unexpected result type
                error_msg = f"Unexpected result type: {type(result)}"
                self.logger.error(f"Unexpected result for {urls[i]}: {error_msg}")
                processed_results.append({
                    'url': urls[i],
                    'title': '',
                    'content': '',
                    'success': False,
                    'error': error_msg
                })

        successful_count = sum(1 for r in processed_results if r.get('success'))
        self.logger.info(f"Web search completed: {successful_count}/{len(urls)} URLs successful")

        # Log failed URLs for debugging
        for result in processed_results:
            if not result.get('success'):
                self.logger.warning(f"Failed URL: {result['url']} - {result.get('error', 'Unknown error')}")

        return processed_results
