import logging
from typing import Dict, Any, Optional
from .router import Router, RuleBasedRouter

logger = logging.getLogger("hybrid")

class HybridQueryEngine:
    def __init__(self, rag_pipeline, rlm_client, router: Optional[Router] = None):
        self.rag = rag_pipeline
        self.rlm = rlm_client
        self.router = router or RuleBasedRouter()

    def query(self, query: str, context_tabs: Dict[str, str], context_text: str, task_id: str = "default") -> Dict[str, Any]:
        """
        Execute a query using the hybrid approach.
        
        Args:
            query: The user's question.
            context_tabs: Dictionary of tab_name -> content (for RLM).
            context_text: Full text or list of chunks (for RAG).
            task_id: Unique ID for the task (used for RAG collection naming).
            
        Returns:
            Dict containing:
            - response: The answer string.
            - source: "rlm" or "rag".
            - route: "broad" or "targeted".
        """
        route = self.router.route(query)
        logger.info(f"Query: '{query}' -> Route: {route}")

        if route == "broad":
            return self._execute_rlm(query, context_tabs, route)
        else:
            return self._execute_rag(query, context_text, task_id, route)

    def _execute_rlm(self, query: str, tabs: Dict[str, str], route: str) -> Dict[str, Any]:
        try:
            response = self.rlm.completion(tabs, query)
            return {
                "response": response,
                "source": "rlm",
                "route": route,
                "error": None
            }
        except Exception as e:
            logger.error(f"RLM Error: {e}")
            return {
                "response": None,
                "source": "rlm",
                "route": route,
                "error": str(e)
            }

    def _execute_rag(self, query: str, text: str, task_id: str, route: str) -> Dict[str, Any]:
        try:
            # collection_name must be unique to avoid crosstalk in tests
            response = self.rag.query(text, query, collection_name=f"hybrid_rag_{task_id}")
            return {
                "response": response,
                "source": "rag",
                "route": route,
                "error": None
            }
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return {
                "response": None,
                "source": "rag",
                "route": route,
                "error": str(e)
            }
