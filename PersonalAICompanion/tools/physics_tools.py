import logging
from sympy import sympify, solve, diff, integrate, symbols, SympifyError, latex

# Set up logging for the physics tool
PHYSICS_LOGGER = logging.getLogger("PhysicsTool")

def solve_physics_problem(expression: str, operation: str, variables: str, solve_for: str = None, wrt: str = None) -> dict:
    """
    Solves symbolic physics and math problems using SymPy.

    Args:
        expression (str): The mathematical expression or equation string. e.g., "x**2 + y - z"
        operation (str): The operation to perform. One of ['solve', 'diff', 'integrate', 'simplify'].
        variables (str): A comma-separated string of all variables in the expression. e.g., "x,y,z"
        solve_for (str, optional): The variable to solve for if operation is 'solve'. Defaults to None.
        wrt (str, optional): "With respect to" variable for differentiation ('diff') or integration ('integrate'). Defaults to None.

    Returns:
        dict: A dictionary containing the status and the result as a LaTeX string.
    """
    PHYSICS_LOGGER.info(
        f"Attempting to perform '{operation}' on '{expression}' with variables '{variables}'")

    try:
        # Define the symbolic variables
        syms = symbols(variables)
        # Safely parse the string into a SymPy expression
        expr = sympify(expression)

        result = None
        if operation == 'solve':
            if not solve_for:
                return {"status": "error", "reason": "The 'solve_for' argument is required for the 'solve' operation."}
            target_symbol = symbols(solve_for)
            solution = solve(expr, target_symbol)
            result = solution
        elif operation == 'diff':
            if not wrt:
                return {"status": "error", "reason": "The 'wrt' argument is required for the 'diff' operation."}
            diff_var = symbols(wrt)
            result = diff(expr, diff_var)
        elif operation == 'integrate':
            if not wrt:
                return {"status": "error", "reason": "The 'wrt' argument is required for the 'integrate' operation."}
            int_var = symbols(wrt)
            result = integrate(expr, int_var)
        elif operation == 'simplify':
            result = expr.simplify()
        else:
            return {"status": "error", "reason": f"Unknown operation: {operation}"}

        # Convert the result to a LaTeX string for clean display
        latex_result = latex(result)
        PHYSICS_LOGGER.info(f"Successfully solved. LaTeX result: {latex_result}")
        return {"status": "success", "result_latex": latex_result}

    except (SympifyError, TypeError, ValueError, Exception) as e:
        error_message = f"Failed during symbolic operation: {str(e)}"
        PHYSICS_LOGGER.error(error_message)
        return {"status": "error", "reason": error_message}
