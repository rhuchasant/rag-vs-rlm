import re
from typing import Protocol

class Router(Protocol):
    def route(self, query: str) -> str:
        """Determines if the query is 'broad' (RLM) or 'targeted' (RAG)."""
        ...

class RuleBasedRouter:
    def __init__(self):
        self.broad_phrases = [
            "list all", "every tab", "include every", "extract all",
            "all tab names", "all section", "all tabs", "section numbers",
        ]

    def route(self, query: str) -> str:
        q = query.lower()
        if any(p in q for p in self.broad_phrases):
            return "broad"
        return "targeted"
