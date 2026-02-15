"""
Minimal probe for external RLM wrapper integration.

This script only validates import/discovery and tries a few common call shapes.
It is safe to run even when the wrapper is not installed.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from typing import Any, Optional


MODULE_CANDIDATES = [
    "rlm",
    "recursive_language_model",
    "rlm_wrapper",
]


def _discover_module() -> tuple[Optional[str], Optional[Any]]:
    for name in MODULE_CANDIDATES:
        try:
            return name, importlib.import_module(name)
        except Exception:
            continue
    return None, None


def _pretty(x: Any) -> str:
    try:
        return json.dumps(x, indent=2, ensure_ascii=True, default=str)
    except Exception:
        return str(x)


def _try_call(
    module: Any,
    query: str,
    context: str,
    backend: str,
    model: Optional[str],
) -> Any:
    last_error: Optional[Exception] = None
    # Try common entrypoints without assuming one specific wrapper API.
    for fn_name in ("run", "ask", "query", "complete", "solve"):
        fn = getattr(module, fn_name, None)
        if callable(fn):
            sig = str(inspect.signature(fn))
            print(f"[info] Trying {fn_name}{sig}")
            try:
                return fn(query=query, context=context)
            except TypeError:
                try:
                    return fn(query, context)
                except Exception:
                    continue
            except Exception:
                continue

    for cls_name in ("RLM", "RLMClient", "Client"):
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        try:
            client = cls()
        except Exception:
            continue
        for method_name in ("run", "ask", "query", "solve"):
            method = getattr(client, method_name, None)
            if callable(method):
                sig = str(inspect.signature(method))
                print(f"[info] Trying {cls_name}.{method_name}{sig}")
                try:
                    return method(query=query, context=context)
                except TypeError:
                    try:
                        return method(query, context)
                    except Exception:
                        continue
                except Exception:
                    continue

    # Official alexzhang13/rlm path:
    # rlm.RLM(...).completion(prompt="...")
    rlm_cls = getattr(module, "RLM", None)
    if rlm_cls is not None:
        backend_kwargs = {}
        if model:
            # rlms OpenAI client expects model_name (not model).
            backend_kwargs["model_name"] = model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            backend_kwargs["api_key"] = openai_api_key
        print(f"[info] Trying RLM(backend={backend!r}).completion(prompt=...)")
        try:
            client = rlm_cls(
                backend=backend,
                backend_kwargs=backend_kwargs,
                max_depth=1,
                max_iterations=8,
                verbose=False,
            )
            prompt = (
                "Use the context below to answer the question.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{query}"
            )
            return client.completion(prompt=prompt)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Wrapper imported, but invocation failed. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe external RLM wrapper installation/API.")
    parser.add_argument("--query", required=True, help="User task/query.")
    parser.add_argument("--context", required=True, help="Context text.")
    parser.add_argument(
        "--backend",
        default="openai",
        help="Wrapper backend (e.g., openai/gemini/anthropic). Default: openai",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", os.getenv("RLM_MODEL")),
        help="Optional model override passed as backend_kwargs.model",
    )
    args = parser.parse_args()

    module_name, module = _discover_module()
    if module is None:
        print("[error] Could not import an RLM wrapper module.")
        print("[hint] Tried:", ", ".join(MODULE_CANDIDATES))
        print(
            "[next] Install the wrapper in this environment, then re-run this probe."
        )
        return 1

    print(f"[ok] Imported wrapper module: {module_name}")
    try:
        result = _try_call(
            module,
            args.query,
            args.context,
            backend=args.backend,
            model=args.model,
        )
        print("[ok] Wrapper returned a response.")
        print(_pretty(result))
        return 0
    except Exception as exc:
        print("[error] Wrapper import succeeded, but invocation failed.")
        print(f"[error] {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
