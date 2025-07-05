"""
Enhanced web search tool implementation that reports both successes and failures
"""

from typing import List, Dict, Any
from utils.custom_web_search import CustomWebSearcher

class WebSearchTool:
    """Custom web search tool that can be called by the LLM"""

    def __init__(self, available_urls: List[str], logger=None):
        self.available_urls = available_urls
        self.logger = logger

    async def execute_search(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the web search tool"""
        try:
            urls_to_search = tool_input.get("urls", [])
            reason = tool_input.get("reason", "")

            if self.logger:
                self.logger.info(f"LLM requested web search for {len(urls_to_search)} URLs. Reason: {reason}")

            # Perform the search
            async with CustomWebSearcher(self.logger) as searcher:
                all_results = await searcher.search_urls(urls_to_search, max_concurrent=3)

            if not all_results:
                return {
                    "type": "text",
                    "text": f"No content could be extracted from any of the {len(urls_to_search)} requested URLs."
                }

            # Separate successful and failed results
            successful_results = [r for r in all_results if r.get('success')]
            failed_results = [r for r in all_results if not r.get('success')]

            # Format successful results
            formatted_results = []
            for result in successful_results:
                formatted_results.append(f"""
URL: {result['url']}
Title: {result['title']}
Content: {result['content']}
""")

            # Build response text
            response_parts = []
            response_parts.append(f"Web search completed for {len(urls_to_search)} requested URLs:")
            response_parts.append(f"✓ Successfully fetched: {len(successful_results)} URLs")

            if failed_results:
                response_parts.append(f"✗ Failed to fetch: {len(failed_results)} URLs")
                for failed in failed_results:
                    response_parts.append(f"  - {failed['url']}: {failed.get('error', 'Unknown error')}")

            if successful_results:
                response_parts.append("\n--- SUCCESSFUL RESULTS ---")
                response_parts.extend(formatted_results)

            response_text = "\n".join(response_parts)

            return {
                "type": "text",
                "text": response_text
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"Web search tool execution failed: {e}")
            return {
                "type": "text",
                "text": f"Web search failed: {str(e)}"
            }

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return the tool definition for the LLM"""
        return {
            "name": "web_search",
            "description": "Search specific URLs from the news articles to get additional content and context. Use this when you need more details from the original sources.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to search. Only URLs from the provided news articles are available."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you want to search these specific URLs"
                    }
                },
                "required": ["urls", "reason"]
            }
        }
