import math
import logging

# Set up logging for the math tool
MATH_LOGGER = logging.getLogger("MathTool")

def solve_math_problem(expression: str) -> dict:
    """
    Safely evaluates a mathematical expression using Python's math library.

    Args:
        expression: A string containing the mathematical expression to solve.
                    e.g., "2 * (3 + 4) / math.sin(math.pi / 2)"

    Returns:
        A dictionary containing the result or an error message.
    """
    MATH_LOGGER.info(f"Attempting to solve expression: {expression}")

    # Create a safe dictionary of allowed names for eval().
    # This includes all functions and constants from the math module.
    safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    safe_dict['abs'] = abs
    safe_dict['round'] = round

    try:
        # Evaluate the expression within the safe context
        result = eval(expression, {"__builtins__": None}, safe_dict)
        MATH_LOGGER.info(f"Successfully solved expression. Result: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        error_message = f"Failed to evaluate expression: {str(e)}"
        MATH_LOGGER.error(error_message)
        return {"status": "error", "reason": error_message}

