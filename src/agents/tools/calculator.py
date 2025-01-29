"""Calculator tool implementation."""
import math
import re

import numexpr
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Calculates mathematical expressions using numexpr. Supports basic arithmetic, mathematical functions, and constants like pi and e."""
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},
                local_dict=local_dict,
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'Calculator("{expression}") raised error: {e}. Please try again with a valid numerical expression'
        )
