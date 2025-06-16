import json


def get_standard_planning_prompt(user_prompt: str, project_state: str) -> str:
    """Returns the standard planning prompt."""
    return f"""
    You are an expert AI project architect. Your task is to break down a user's request into a series of discrete, numbered steps.
    Provide the output as a JSON array of objects, where each object has "id" (integer), "task" (string), and "status" (string, default 'pending').

    User Request: "{user_prompt}"

    Current Project State:
    {project_state}

    Generate the JSON plan now.
    """


def get_deep_reasoning_prompt(user_prompt: str, project_state: str) -> str:
    """
    Returns the deep reasoning planning prompt for ultra-deep thinking,
    inspired by the user's provided template.
    """
    return f"""
    You are to engage in ultra-deep thinking mode with extreme rigor and multi-angle verification.

    First, outline the user's request and break it down into high-level subtasks.
    For each subtask, explore multiple implementation perspectives, even those that seem initially irrelevant.
    Purposefully attempt to challenge your own assumptions at every step.

    Critically review your logic, scrutinize assumptions, and explicitly call out uncertainties.
    Your final output must be a detailed, step-by-step plan formatted as a JSON array of objects.
    Each object must have "id" (integer), "task" (string), "status" (string, default 'pending'), and a "reasoning" (string) field explaining why the step is necessary and what assumptions it makes.

    <task>
    User Request: "{user_prompt}"

    Current Project State:
    {project_state}
    </task>

    Generate the JSON plan now, ensuring every step includes detailed reasoning.
    """

def get_self_correction_prompt(original_prompt: str, failed_step_id: int, current_plan: list[dict], scratchpad: list[dict], project_state: str) -> str:
    """
    Returns a prompt designed to guide the agent in self-correction after a failed execution step.
    """
    plan_str = json.dumps(current_plan, indent=2)
    scratchpad_str = json.dumps(scratchpad, indent=2) # Assuming scratchpad contains dicts

    return f"""
    A previous attempt to execute a plan encountered a failure. Your task is to analyze the failure and propose a revised plan to overcome it.

    User's Original Request: "{original_prompt}"

    Failed Step ID: {failed_step_id}

    Current Plan (including the failed step and subsequent steps):
    ```json
    {plan_str}
    ```

    Scratchpad (recent thoughts and tool outputs leading up to the failure):
    ```json
    {scratchpad_str}
    ```

    Current Project State:
    {project_state}

    Based on the above information, critically analyze what went wrong in step {failed_step_id}.
    Consider the following:
    1.  Was the original plan flawed for this step?
    2.  Was there an issue with the execution of the tools?
    3.  Is the project state different than anticipated?

    Propose a revised plan starting from step {failed_step_id}. The revised plan should be a JSON array of objects, identical in structure to the original plan format (each object with "id", "task", "status", and if deep reasoning was used, "reasoning"). Ensure the IDs are sequential and start from {failed_step_id}.

    Generate the REVISED JSON plan now.
    """
