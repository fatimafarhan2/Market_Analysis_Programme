# debug_tool_annotations.py
"""
Improved diagnostic for annotation evaluation on tool functions.
Handles both plain functions and wrapped tool objects (StructuredTool etc).
Run from project root with the same venv you use for the project.
"""

import sys
import traceback
from typing import get_type_hints
import inspect

# --- Update these imports to match how your main script imports tools ---
try:
    # import the way your pipeline does
    from tools import swot_analysis as swot_tool
    from market_tool import deep_market_analysis
    from reviews_summary_tool import fetch_reviews_summary
except Exception:
    print("ERROR importing tool functions. Update import paths at top of this file.")
    traceback.print_exc()
    sys.exit(2)

tool_list = [swot_tool, deep_market_analysis, fetch_reviews_summary]

def _resolve_callable(obj):
    """
    If obj is a wrapper (StructuredTool or similar), try to extract the original function.
    Common attributes to try: 'func', '_func', '__wrapped__'.
    Otherwise return the object itself.
    """
    for attr in ("func", "_func", "__wrapped__", "__call__"):
        if hasattr(obj, attr):
            candidate = getattr(obj, attr)
            # If attribute is a function, return it; if it's a bound method, return its function
            if inspect.isfunction(candidate) or inspect.ismethod(candidate):
                return candidate
    # fallback: if object itself is a function/method, return it
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return obj
    # nothing found
    return None

def readable_name(obj):
    """Return a human-friendly name for the object."""
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "name"):
        return getattr(obj, "name")
    # structured tool repr often includes class name
    return f"{obj.__class__.__name__}: {repr(obj)}"

def inspect_tool(obj):
    print("\n" + "-"*60)
    print("Inspecting object:", readable_name(obj))
    # try to resolve underlying function
    target = _resolve_callable(obj)
    if target is None:
        print("Could not locate a plain Python function inside this object.")
        print("Object type:", type(obj))
        print("repr:", repr(obj)[:400])
        print("Skipping annotation evaluation for this object.\n"
              "If this is a LangChain StructuredTool, open its original function module and add missing typing imports there.")
        return False

    print("Resolved underlying callable:", readable_name(target), "from module:", getattr(target, "__module__", "<unknown>"))
    try:
        hints = get_type_hints(target, include_extras=True)
        print("OK — type hints evaluated successfully.")
        if hints:
            print("Type hints:", hints)
        else:
            print("No annotations / empty hints.")
    except Exception:
        print("ERROR evaluating annotations for the callable:")
        traceback.print_exc()
        # show a helpful snapshot of the module where the function is defined
        modname = getattr(target, "__module__", None)
        if modname and modname in sys.modules:
            mod = sys.modules[modname]
            keys = list(mod.__dict__.keys())
            print("\nModule globals keys (first 120):")
            for k in keys[:120]:
                print(" ", k)
            # quick check whether Annotated exists in that module globals
            print("\nQuick check: 'Annotated' in module globals? ->", "Annotated" in mod.__dict__)
        else:
            print("Could not find module object for", modname)
        return False
    return True

def main():
    all_ok = True
    for obj in tool_list:
        ok = inspect_tool(obj)
        if not ok:
            all_ok = False
    if all_ok:
        print("\nEverything looks fine: no annotation evaluation errors detected.")
    else:
        print(
            "\nOne or more tools failed annotation evaluation or could not be resolved."
            "\nIf a tool was a wrapped object (StructuredTool), open the original module where the function is defined"
            " and add 'from typing import Annotated' and other needed typing names (Literal, TypedDict, etc.) at the top."
        )
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
