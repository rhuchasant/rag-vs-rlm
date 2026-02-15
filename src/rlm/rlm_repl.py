"""
RLM client - v2 with stricter enforcement of code-before-FINAL rule.
"""

import logging
import time
from typing import Optional, Dict, Any

from .repl_env import REPLEnv
from .prompts import build_messages, _user_prompt
from .utils import (
    find_code_blocks,
    find_final_answer,
    format_execution_result,
    sanitize_numeric_result,
    extract_number_from_text,
    is_likely_code,
)
from ..llm_client import chat_completion, get_default_models

log = logging.getLogger("rlm")


class RLMClient:
    """RLM client with strict code execution enforcement."""

    def __init__(
        self,
        root_model: Optional[str] = None,
        recursive_model: Optional[str] = None,
        max_iterations: int = 10,
        summarize_execution: bool = True,
    ):
        """Initialize RLM client."""
        if root_model is None or recursive_model is None:
            defaults = get_default_models()
            root_model = root_model or defaults[0]
            recursive_model = recursive_model or defaults[1]
        
        self.root_model = root_model
        self.recursive_model = recursive_model
        self.max_iterations = max_iterations
        self.summarize_execution = summarize_execution
        
        log.info(f"[RLM] Initialized with root={root_model}, recursive={recursive_model}")

    def completion(
        self,
        context: str,
        query: str,
        context_json: Optional[Dict] = None,
    ) -> str:
        """Complete RLM workflow with strict code execution enforcement.
        context: str or dict. If dict (e.g. multi-tab), passed as context_json.
        """
        log.info(f"[RLM] Starting query: {query[:80]}...")
        if isinstance(context, dict):
            context_json = context_json or context
            context_str = None
            log.info(f"[RLM] Context: dict with {len(context_json)} keys")
        else:
            context_str = context if not context_json else None
            log.info(f"[RLM] Context: {len(context):,} chars")
        
        start_time = time.time()
        
        # Initialize REPL environment
        repl_env = REPLEnv(
            context_str=context_str,
            context_json=context_json,
            recursive_model=self.recursive_model,
        )
        
        # Build initial messages
        messages = build_messages(query)
        iteration = 0
        final_answer = None
        has_executed_code = False  # Track if any code was executed
        
        while iteration < self.max_iterations:
            iteration += 1
            log.info(f"[RLM] === Iteration {iteration}/{self.max_iterations} ===")
            
            # Get response from root model
            try:
                response = chat_completion(
                    messages=messages,
                    model=self.root_model,
                    max_tokens=4096,
                )
            except Exception as e:
                log.error(f"[RLM] Root model error: {e}")
                return f"Error: {str(e)}"
            
            log.info(f"[RLM] Root model response: {len(response)} chars")
            messages.append({"role": "assistant", "content": response})
            
            # Extract and execute code blocks FIRST
            code_blocks = find_code_blocks(response)
            
            if code_blocks:
                # Execute all code blocks
                execution_outputs = []
                for code_idx, code in enumerate(code_blocks):
                    log.info(f"[RLM] Executing code block {code_idx + 1}/{len(code_blocks)}")
                    log.debug(f"[RLM] Code:\n{code[:200]}...")
                    
                    exec_result = repl_env.code_execution(code)
                    
                    if exec_result.success:
                        has_executed_code = True
                        log.info(f"[RLM] Execution successful ({exec_result.execution_time:.2f}s)")
                        
                        if self.summarize_execution:
                            summary = format_execution_result(
                                exec_result.stdout,
                                exec_result.stderr,
                                exec_result.locals,
                            )
                            execution_outputs.append(summary)
                        else:
                            output = []
                            if exec_result.stdout:
                                output.append(f"stdout:\n{exec_result.stdout}")
                            if exec_result.stderr:
                                output.append(f"stderr:\n{exec_result.stderr}")
                            execution_outputs.append("\n".join(output) if output else "Executed successfully")
                    else:
                        log.error(f"[RLM] Execution error: {exec_result.error}")
                        execution_outputs.append(f"ERROR: {exec_result.error}")
                
                # Add execution results to conversation
                exec_summary = "\n\n".join(execution_outputs)
                messages.append({
                    "role": "user",
                    "content": f"REPL execution results:\n{exec_summary}\n\n"
                              f"Available variables: {list(repl_env.locals.keys())}\n\n"
                              f"Continue or provide final answer with FINAL_VAR(\"variable_name\")."
                })
            
            # NOW check for FINAL answer marker
            final_marker = find_final_answer(response)
            if final_marker:
                kind, content = final_marker
                log.info(f"[RLM] Found {kind}({content})")
                
                # Check if code was executed
                if not has_executed_code and iteration == 1:
                    # First iteration, no code yet - reject FINAL
                    log.warning(f"[RLM] Model tried to use {kind} without executing code first")
                    messages.append({
                        "role": "user",
                        "content": f"ERROR: You must write and execute code in ```repl blocks BEFORE using {kind}.\n\n"
                                  f"Write code to:\n"
                                  f"1. Process the context\n"
                                  f"2. Store the result in a variable\n"
                                  f"3. THEN use FINAL_VAR(\"variable_name\")"
                    })
                    continue
                
                if kind == "FINAL":
                    # Check if it's a variable name (bad) or literal value (ok)
                    if content in repl_env.locals:
                        # It's a variable name - should have used FINAL_VAR
                        log.warning(f"[RLM] Model used FINAL({content}) but {content} is a variable")
                        messages.append({
                            "role": "user",
                            "content": f"ERROR: '{content}' is a variable. Use FINAL_VAR(\"{content}\") instead of FINAL({content}).\n\n"
                                      f"FINAL(x) returns the literal text 'x'\n"
                                      f"FINAL_VAR(\"x\") returns the value of variable x"
                        })
                        continue
                    else:
                        # Direct literal answer
                        final_answer = content
                        break
                
                elif kind == "FINAL_VAR":
                    # Extract from REPL variable
                    var_value = repl_env.get_variable(content)
                    if var_value is not None:
                        final_answer = str(var_value)
                        log.info(f"[RLM] Extracted {content} = {final_answer}")
                        break
                    else:
                        # Variable not found - ask model to fix
                        log.warning(f"[RLM] Variable '{content}' not found in REPL")
                        available_vars = list(repl_env.locals.keys())
                        messages.append({
                            "role": "user",
                            "content": f"ERROR: Variable '{content}' does not exist.\n\n"
                                      f"Available variables: {available_vars}\n\n"
                                      f"You must create the variable first:\n"
                                      f"```repl\n"
                                      f"{content} = your_computed_value\n"
                                      f"```\n\n"
                                      f"THEN use FINAL_VAR(\"{content}\")"
                        })
                        continue
            
            # No code and no FINAL marker
            if not code_blocks and not final_marker:
                if iteration >= self.max_iterations - 1:
                    # Last iteration - force answer from variables if possible
                    log.warning("[RLM] Max iterations reached, extracting from variables")
                    final_answer = self._extract_fallback_answer(response, repl_env)
                    break
                else:
                    # Ask for code or final answer
                    messages.append({
                        "role": "user",
                        "content": "Please write code in ```repl blocks to analyze the context, "
                                  "or provide your final answer with FINAL_VAR(\"variable_name\") if you already have the result."
                    })
                    continue
        
        # Post-process final answer
        if final_answer is None:
            log.error("[RLM] No final answer found after max iterations")
            final_answer = self._extract_fallback_answer(response, repl_env)
        
        # Clean up the answer
        if final_answer:
            final_answer = str(final_answer).strip()
            
            # Sanitize numeric results
            if final_answer.replace("-", "").replace(".", "").isdigit():
                sanitized = sanitize_numeric_result(final_answer)
                if sanitized:
                    final_answer = sanitized
                    log.info(f"[RLM] Sanitized result: {final_answer}")
        
        elapsed = time.time() - start_time
        log.info(f"[RLM] Completed in {elapsed:.1f}s, {iteration} iterations")
        log.info(f"[RLM] Final answer: {final_answer}")
        
        debug_info = repl_env.get_debug_info()
        log.info(f"[RLM] Sub-LLM calls: {debug_info['sub_llm_calls']}")
        
        return final_answer if final_answer else "Error: Could not extract answer"

    def _extract_fallback_answer(self, last_response: str, repl_env: REPLEnv) -> Optional[str]:
        """Fallback: extract answer from REPL variables."""
        log.warning("[RLM] Using fallback answer extraction")
        
        # Check common variable names in order of preference
        for var_name in ['final_answer', 'answer', 'result', 'total', 'count']:
            value = repl_env.get_variable(var_name)
            if value is not None:
                log.info(f"[RLM] Found answer in variable '{var_name}': {value}")
                return str(value)
        
        # Try to extract from response text
        if not is_likely_code(last_response):
            num = extract_number_from_text(last_response)
            if num:
                log.info(f"[RLM] Extracted number from response: {num}")
                return num
        
        return None