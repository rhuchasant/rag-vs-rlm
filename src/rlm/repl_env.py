"""
REPL environment for RLM - stores context as variable, allows code execution with sub-LLM calls.
Based on MIT RLM paper (Zhang et al., 2025) and alexzhang13/rlm-minimal.
Enhanced with robust counting, filtering, and aggregation utilities.
"""

import logging
import os
import re
import time
import sys
import io
import json
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

log = logging.getLogger("rlm")


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    success: bool = True
    error: Optional[str] = None


class SubLLM:
    """Sub-LLM for recursive calls within REPL. Uses smaller/cheaper model."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.call_count = 0

    def completion(self, prompt) -> str:
        from ..llm_client import chat_completion
        
        self.call_count += 1
        
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        elif isinstance(prompt, list):
            messages = prompt
            preview = str(prompt)[:80] + "..."
        else:
            messages = [{"role": "user", "content": str(prompt)}]
            preview = str(prompt)[:80] + "..."
        
        try:
            t0 = time.time()
            out = chat_completion(messages=messages, model=self.model, max_tokens=4096)
            elapsed = time.time() - t0
            log.info(f"[RLM] sub-LLM call #{self.call_count} done ({elapsed:.1f}s) prompt: {preview}")
            return out
        except Exception as e:
            log.warning(f"[RLM] sub-LLM error: {e}")
            return f"Error: {str(e)}"


class REPLEnv:
    """
    REPL environment where context is stored as a variable.
    Model writes code to inspect, decompose, and recursively call sub-LLM.
    Enhanced with robust helper functions for counting and aggregation tasks.
    """

    def __init__(
        self,
        context_str: Optional[str] = None,
        context_json: Optional[dict] = None,
        recursive_model: str = "gpt-4o-mini",
    ):
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self.sub_llm = SubLLM(model=recursive_model)
        self._lock = threading.Lock()
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.debug_info = {
            'sub_queries': [],
            'execution_steps': []
        }

        # ===== HELPER FUNCTIONS FOR CODE EXECUTION =====
        
        def llm_query(prompt: Union[str, List]) -> str:
            """
            Query sub-LLM with a prompt. Returns string response.
            Use for semantic understanding or complex reasoning tasks.
            """
            result = self.sub_llm.completion(prompt)
            self.debug_info['sub_queries'].append({
                'prompt': str(prompt)[:200],
                'response': result[:200]
            })
            return result

        def llm_query_count(prompt: Union[str, List], context_chunk: Optional[str] = None) -> str:
            """
            Query sub-LLM specifically for counting tasks.
            Appends instructions to return only a number.
            
            Args:
                prompt: The counting question
                context_chunk: Optional specific text to count in
            
            Returns:
                String response (should be a number)
            """
            if context_chunk:
                full_prompt = f"""Context:
{context_chunk}

Question: {prompt}

Reply with ONLY the number. If the count is zero, respond with 0. No explanation."""
            else:
                full_prompt = f"""{prompt}

Reply with ONLY the number. If the count is zero, respond with 0. No explanation."""
            
            result = self.sub_llm.completion(full_prompt)
            self.debug_info['sub_queries'].append({
                'type': 'count',
                'prompt': str(prompt)[:200],
                'response': result
            })
            return result

        def parse_count(response: Any) -> int:
            """
            Parse sub-LLM response to extract an integer count.
            Returns 0 if not parseable.
            
            Examples:
                "42" -> 42
                "The count is 15" -> 15
                "zero" -> 0
                "" -> 0
            """
            if response is None:
                return 0
            
            s = str(response).strip().lower()
            if not s:
                return 0
            
            # Handle word numbers
            word_to_num = {
                'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            if s in word_to_num:
                return word_to_num[s]
            
            # Extract first number from response
            nums = re.findall(r'\b\d+\b', s)
            if nums:
                return int(nums[0])
            
            return 0

        def aggregate_counts(responses: List[Any]) -> int:
            """
            Parse list of sub-LLM responses and SUM them.
            Use for aggregating counts across document chunks.
            
            Example:
                responses = ["5", "12", "8"]
                result = aggregate_counts(responses)  # Returns 25
            """
            total = 0
            for r in responses:
                count = parse_count(r)
                total += count
            return total

        def count_pattern(pattern: str, text: Optional[str] = None, 
                         case_sensitive: bool = False) -> int:
            """
            Count occurrences of a regex pattern in text.
            If text is None, searches in the entire context.
            
            Args:
                pattern: Regex pattern to search for
                text: Text to search in (or None for full context)
                case_sensitive: Whether to match case
            
            Returns:
                Number of matches
            """
            if text is None:
                text = self.locals.get('context', '')
            
            flags = 0 if case_sensitive else re.IGNORECASE
            matches = re.findall(pattern, str(text), flags=flags)
            return len(matches)

        def filter_context(keyword: str, case_sensitive: bool = False) -> str:
            """
            Filter context to only lines containing keyword.
            Useful for narrowing down context before sub-LLM queries.
            
            Args:
                keyword: Word or phrase to filter by
                case_sensitive: Whether to match case
            
            Returns:
                Filtered text containing only matching lines
            """
            context = self.locals.get('context', '')
            if not context:
                return ""
            
            lines = str(context).split('\n')
            keyword_cmp = keyword if case_sensitive else keyword.lower()
            
            filtered_lines = []
            for line in lines:
                line_cmp = line if case_sensitive else line.lower()
                if keyword_cmp in line_cmp:
                    filtered_lines.append(line)
            
            return '\n'.join(filtered_lines)

        def chunk_context(max_chars: int = 2000) -> List[str]:
            """
            Split context into manageable chunks for sub-LLM processing.
            Useful for processing long documents in parallel.
            
            Args:
                max_chars: Maximum characters per chunk
            
            Returns:
                List of text chunks
            """
            context = self.locals.get('context', '')
            if not context:
                return []
            
            text = str(context)
            chunks = []
            
            # Try to split on paragraph boundaries
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= max_chars:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraphs are too long, fall back to character splitting
            if not chunks or any(len(c) > max_chars for c in chunks):
                chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
            
            return chunks

        def extract_numbers(text: Optional[str] = None) -> List[int]:
            """
            Extract all integers from text.
            If text is None, extracts from full context.
            
            Returns:
                List of integers found
            """
            if text is None:
                text = self.locals.get('context', '')
            
            numbers = re.findall(r'-?\b\d+\b', str(text))
            return [int(n) for n in numbers]

        def sanitize_result(value: Any) -> Union[int, float, str]:
            """
            Clean up result value for final answer.
            Handles common issues like duplicated digits, string concatenation.
            
            Args:
                value: Raw result value
            
            Returns:
                Cleaned result
            """
            if value is None:
                return 0
            
            # Handle strings
            if isinstance(value, str):
                val_stripped = value.strip()
                
                # Check for duplicated digits (e.g., "7373" -> "73")
                if val_stripped.isdigit() and len(val_stripped) % 2 == 0:
                    half = len(val_stripped) // 2
                    first_half = val_stripped[:half]
                    second_half = val_stripped[half:]
                    if first_half == second_half:
                        val_stripped = first_half
                
                # Try to convert to number
                try:
                    if '.' in val_stripped:
                        return float(val_stripped)
                    return int(val_stripped)
                except ValueError:
                    # Extract first number if present
                    match = re.search(r'-?\d+\.?\d*', val_stripped)
                    if match:
                        num_str = match.group()
                        return float(num_str) if '.' in num_str else int(num_str)
                    return val_stripped
            
            # Handle numeric types
            if isinstance(value, (int, float)):
                return value
            
            # Handle lists - sum if all numeric
            if isinstance(value, list):
                if all(isinstance(x, (int, float)) for x in value):
                    return sum(value)
                return str(value)
            
            return str(value)

        def final_var(name: str) -> str:
            """
            Get value of a variable by name (for FINAL_VAR marker).
            Used by RLM client to extract final answer.
            """
            name = name.strip().strip('"').strip("'").strip()
            if name in self.locals:
                return str(self.locals[name])
            return f"Error: Variable '{name}' not found"

        # ===== SET UP GLOBALS FOR CODE EXECUTION =====
        
        self.globals = {
            # Builtins
            "__builtins__": __builtins__,
            
            # LLM query functions
            "llm_query": llm_query,
            "llm_query_count": llm_query_count,
            
            # Parsing and aggregation
            "parse_count": parse_count,
            "aggregate_counts": aggregate_counts,
            
            # Text processing
            "count_pattern": count_pattern,
            "filter_context": filter_context,
            "chunk_context": chunk_context,
            "extract_numbers": extract_numbers,
            
            # Result handling
            "sanitize_result": sanitize_result,
            "FINAL_VAR": final_var,
            
            # Standard Python functions
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            
            # Modules
            "re": __import__("re"),
            "json": __import__("json"),
            "math": __import__("math"),
        }
        
        self.locals = {}

        # Load context
        if context_str is not None:
            self._load_context_str(context_str)
        if context_json is not None:
            self._load_context_json(context_json)

    def _load_context_str(self, text: str):
        """Load context from string."""
        path = os.path.join(self.temp_dir, "context.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.locals["context"] = text
        log.info(f"[RLM] Loaded context: {len(text)} chars")

    def _load_context_json(self, data: dict):
        """Load context from JSON."""
        path = os.path.join(self.temp_dir, "context.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.locals["context"] = data
        log.info(f"[RLM] Loaded context: JSON with {len(data)} keys")

    @contextmanager
    def _capture_output(self):
        """Capture stdout and stderr during code execution."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            out_buf, err_buf = io.StringIO(), io.StringIO()
            sys.stdout, sys.stderr = out_buf, err_buf
            try:
                yield out_buf, err_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    def code_execution(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment.
        Returns result with stdout, stderr, locals, and timing.
        """
        start = time.time()
        error_msg = None
        success = True
        
        with self._capture_output() as (out_buf, err_buf):
            try:
                exec(code, self.globals, self.locals)
                log.info(f"[RLM] Code executed successfully ({time.time() - start:.2f}s)")
            except Exception as e:
                error_msg = str(e)
                err_buf.write(f"Execution error: {error_msg}\n")
                log.error(f"[RLM] Code execution error: {error_msg}")
                success = False
        
        elapsed = time.time() - start
        
        self.debug_info['execution_steps'].append({
            'code': code[:200],
            'time': elapsed,
            'success': success,
            'error': error_msg
        })
        
        return REPLResult(
            stdout=out_buf.getvalue(),
            stderr=err_buf.getvalue(),
            locals=dict(self.locals),
            execution_time=elapsed,
            success=success,
            error=error_msg,
        )

    def get_variable(self, name: str) -> Any:
        """Get value of a variable from locals."""
        return self.locals.get(name)

    def get_debug_info(self) -> dict:
        """Get debug information about sub-queries and execution."""
        return {
            **self.debug_info,
            'sub_llm_calls': self.sub_llm.call_count,
            'variables': list(self.locals.keys())
        }

    def __del__(self):
        """Clean up temporary directory."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass