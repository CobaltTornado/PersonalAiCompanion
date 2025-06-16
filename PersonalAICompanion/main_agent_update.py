import os
import re
import json
import logging
import asyncio
import threading
import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai.protos as protos
from abc import ABC, abstractmethod

# Import our modular tools and the new prompt templates
from tools.git_tools import git_commit, git_status
from tools.file_system_tools import create_file, read_file, update_file, get_fs_state_str
from tools.math_tools import solve_math_problem
from tools.prompt_templates import get_standard_planning_prompt, get_deep_reasoning_prompt

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
AGENT_LOGGER = logging.getLogger("ChiefArchitectAgent")


class ProgressManager:
    """Manages WebSocket connections and broadcasts messages in a thread-safe way."""

    def __init__(self):
        self.connections = set()
        self._lock = threading.Lock()
        AGENT_LOGGER.info("Progress Manager initialized.")

    def add_connection(self, ws):
        with self._lock:
            self.connections.add(ws)
        AGENT_LOGGER.info(f"Client connected. Total connections: {len(self.connections)}")

    def remove_connection(self, ws):
        with self._lock:
            self.connections.discard(ws)
        AGENT_LOGGER.info(f"Client disconnected. Total connections: {len(self.connections)}")

    def _broadcast_sync(self, data: dict):
        message = json.dumps(data)
        closed_connections = set()
        with self._lock:
            for ws in list(self.connections):
                try:
                    ws.send(message)
                except Exception as e:
                    AGENT_LOGGER.warning(f"Failed to send to a client, removing: {e}")
                    closed_connections.add(ws)
        if closed_connections:
            with self._lock:
                self.connections.difference_update(closed_connections)

    async def broadcast(self, data_type: str, payload: any):
        data = {"type": data_type, "payload": payload}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._broadcast_sync, data)


class TaskExecutionContext:
    """Holds all state for a single, complex task execution."""

    def __init__(self, original_prompt: str, progress_manager: ProgressManager):
        self.original_prompt = original_prompt
        self.plan: list[dict] = []
        self.scratchpad: list[dict] = []
        self.progress_manager = progress_manager
        AGENT_LOGGER.info(f"TaskExecutionContext created for prompt: '{original_prompt[:50]}...'")

    async def add_to_scratchpad(self, entry_type: str, detail: str):
        self.scratchpad.append({'type': entry_type, 'detail': detail})
        AGENT_LOGGER.info(f"Scratchpad Updated: {entry_type} - {detail}")
        await self.progress_manager.broadcast("log", f"Scratchpad: {entry_type} - {detail}")

    async def set_plan(self, plan_list: list[dict]):
        self.plan = plan_list
        await self.add_to_scratchpad('INITIAL_PLAN', f'Plan with {len(plan_list)} steps generated.')
        await self.progress_manager.broadcast("plan", self.plan)

    async def update_step_status(self, step_id: int, status: str, detail: str | None = None):
        for step in self.plan:
            if step.get('id') == step_id:
                step['status'] = status
                if detail: step['detail'] = detail
                await self.add_to_scratchpad('STEP_STATUS_UPDATE', f"Step {step_id} is now {status}.")
                await self.progress_manager.broadcast("plan", self.plan)
                return
        AGENT_LOGGER.warning(f"Attempted to update status for non-existent step_id: {step_id}")


class BaseModeHandler(ABC):
    """Abstract base class for handling different agent modes."""

    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager

    @abstractmethod
    async def handle(self, prompt: str, task_context: TaskExecutionContext | None = None, deep_reasoning: bool = False):
        pass


class ChatModeHandler(BaseModeHandler):
    """Handles simple chat interactions."""

    def __init__(self, progress_manager: ProgressManager):
        super().__init__(progress_manager)
        self.text_chat_model = genai.GenerativeModel(os.environ.get("FLASH_MODEL_NAME", "gemini-1.5-flash-latest"))

    async def handle(self, prompt: str, task_context: TaskExecutionContext | None = None, deep_reasoning: bool = False):
        await self.progress_manager.broadcast("log", "Engaging in chat mode...")
        chat_prompt = f"You are a helpful AI assistant. A user said: '{prompt}'. Respond in a friendly, direct manner."

        response_stream = await self.text_chat_model.generate_content_async(chat_prompt, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
            except (ValueError, IndexError):
                AGENT_LOGGER.debug("Received an empty stream chunk.")
                pass
        await self.progress_manager.broadcast("final_result", "Chat interaction complete.")


class MathModeHandler(BaseModeHandler):
    """Handles mathematical problem-solving."""

    def __init__(self, progress_manager: ProgressManager, chat_model):
        super().__init__(progress_manager)
        self.chat_model = chat_model

    async def handle(self, prompt: str, task_context: TaskExecutionContext | None = None, deep_reasoning: bool = False):
        await self.progress_manager.broadcast("log", "Engaging in Math mode...")

        math_prompt = (
            "You are a mathematics expert. A user has a math problem. "
            "Your task is to provide a step-by-step solution. "
            "Format your response as a JSON object with a single key 'solution'. "
            "The value of 'solution' should be an array of objects, where each object has a 'type' ('text' or 'latex') and 'content' (the string). "
            "Ensure the LaTeX is correctly formatted for rendering. "
            f"\n\nUser's problem: '{prompt}'"
        )

        response = await self.chat_model.generate_content_async(math_prompt)

        try:
            solution_data = json.loads(response.text)
            solution_steps = solution_data.get("solution", [])

            for step in solution_steps:
                content_type = step.get("type")
                content = step.get("content")
                if content_type == "text":
                    await self.progress_manager.broadcast("chat_chunk", content)
                elif content_type == "latex":
                    await self.progress_manager.broadcast("latex_canvas", content)
        except (json.JSONDecodeError, AttributeError) as e:
            AGENT_LOGGER.error(f"Failed to parse math solution JSON: {e}")
            await self.progress_manager.broadcast("log",
                                                  "I had trouble formatting the math solution. Please try again.")

        await self.progress_manager.broadcast("final_result", "Math problem solved.")


class TaskModeHandler(BaseModeHandler):
    """Handles complex multi-step tasks involving planning and tool execution."""

    def __init__(self, progress_manager: ProgressManager, planner_model, executor_model, tools: dict, mark_step_complete_func):
        super().__init__(progress_manager)
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.tools = tools
        self._mark_step_complete_func = mark_step_complete_func # Store the instance method

    async def _perform_planning(self, task_context: TaskExecutionContext, deep_reasoning: bool):
        await self.progress_manager.broadcast("log",
                                              f"Phase 1: Planning (Mode: {'Deep Reasoning' if deep_reasoning else 'Standard'})...")
        git_status_output = git_status().get('output', 'Not available.') # Call once
        project_state = f"--- File System ---\n{get_fs_state_str()}\n--- Git Status ---\n{git_status_output}"
        prompt = get_deep_reasoning_prompt(task_context.original_prompt,
                                           project_state) if deep_reasoning else get_standard_planning_prompt(
            task_context.original_prompt, project_state)

        response_stream = await self.planner_model.generate_content_async(prompt, stream=True)
        full_response_text = ""
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    text_part = chunk.parts[0].text
                    full_response_text += text_part
                    await self.progress_manager.broadcast("reasoning_chunk", text_part)
            except (ValueError, IndexError):
                AGENT_LOGGER.debug("Received an empty stream chunk during planning.")
                pass

        json_match = re.search(r'```json\n(.*?)\n```', full_response_text, re.DOTALL) or re.search(r'(\[.*\])',
                                                                                                   full_response_text,
                                                                                                   re.DOTALL)
        if json_match:
            try:
                await task_context.set_plan(json.loads(json_match.group(1)))
            except json.JSONDecodeError as e:
                AGENT_LOGGER.error(f"Failed to parse JSON plan from LLM. Raw response: {full_response_text[:500]}... Error: {e}")
                raise ValueError("LLM returned malformed JSON plan.")
        else:
            AGENT_LOGGER.error(f"LLM did not return a valid JSON plan. Raw response: {full_response_text[:500]}...")
            raise ValueError("LLM did not return a valid JSON plan from its reasoning stream.")


    async def _perform_execution(self, task_context: TaskExecutionContext):
        await self.progress_manager.broadcast("log", "Phase 2: Execution...")
        chat_session = self.executor_model.start_chat()

        for step in task_context.plan:
            if step.get('status') == 'completed': continue
            await task_context.update_step_status(step['id'], 'in_progress')
            git_status_output = git_status().get('output', 'Not available.') # Call once
            project_state = f"--- File System ---\n{get_fs_state_str()}\n--- Git Status ---\n{git_status_output}"
            next_message = f"Executing Step {step['id']}: {step['task']}\nUse tools to complete this. Call 'mark_step_complete' when done.\nState:\n{project_state}"

            step_completed = False
            for _ in range(7):  # Max turns per step
                response = await chat_session.send_message_async(next_message)
                if not response.candidates:
                    await self.progress_manager.broadcast("log", "Agent returned an empty response.")
                    break

                try:
                    text_part = response.text
                except (ValueError, IndexError):
                    text_part = None

                function_calls = [p.function_call for p in response.parts if hasattr(p, 'function_call')]

                if text_part: await self.progress_manager.broadcast("log", f"Agent thought: {text_part}")
                if not function_calls: break

                tool_results = []
                for fc in function_calls:
                    tool_name = fc.name
                    args = dict(fc.args) if fc.args else {}

                    # Mark step complete needs to use the function passed in init
                    if tool_name == 'mark_step_complete':
                        result = await self._mark_step_complete_func(step_id=int(args['step_id']), detail=args.get('detail', "Completed successfully."))
                        step_completed = True
                        break

                    if tool_name in self.tools:
                        await task_context.add_to_scratchpad('TOOL_REQUEST', f"Calling {tool_name} with args {args}")
                        result = self.tools[tool_name](**args)
                    else:
                        result = {"status": "error", "reason": f"Tool '{tool_name}' not found."}
                    tool_results.append(
                        protos.Part(function_response=protos.FunctionResponse(name=tool_name, response=result)))

                if step_completed: break
                next_message = tool_results

            if not step_completed:
                await task_context.update_step_status(step['id'], 'failed', 'Agent failed to complete step.')
                raise Exception(f"Step {step['id']} failed to complete.")

    async def _generate_final_response(self, task_context: TaskExecutionContext):
        await self.progress_manager.broadcast("log", "Generating final response...")
        plan_summary = "\n".join([f"- Step {s['id']} ({s['status']}): {s['task']}" for s in task_context.plan])
        summary_prompt = f"You just completed this request: '{task_context.original_prompt}'. You executed this plan:\n{plan_summary}\n\nProvide a friendly, conversational response summarizing what you did."

        text_chat_model = genai.GenerativeModel(os.environ.get("FLASH_MODEL_NAME", "gemini-1.5-flash-latest"))
        response_stream = await text_chat_model.generate_content_async(summary_prompt, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
            except (ValueError, IndexError):
                AGENT_LOGGER.debug("Received an empty stream chunk during final response.")
                pass

    async def handle(self, prompt: str, task_context: TaskExecutionContext, deep_reasoning: bool = False):
        try:
            await self._perform_planning(task_context, deep_reasoning)
            await self._perform_execution(task_context)
            await self._generate_final_response(task_context)
            await self.progress_manager.broadcast("final_result", "Task completed successfully.")
        except Exception as e:
            AGENT_LOGGER.error(f"Task execution failed for prompt '{prompt}': {e}", exc_info=True)
            await self.progress_manager.broadcast("log", f"ERROR: Task failed - {e}")
            await self.progress_manager.broadcast("final_result", f"The agent failed to complete the task. Error: {e}")


class ChiefArchitectAgent:
    def __init__(self, progress_manager: ProgressManager):
        AGENT_LOGGER.info("Initializing ChiefArchitectAgent instance...")
        load_dotenv()
        api_key = os.getenv("API_KEY")
        if not api_key: raise ValueError("API_KEY not found.")
        genai.configure(api_key=api_key)

        self.progress_manager = progress_manager
        self.current_task_context: TaskExecutionContext | None = None

        pro_model_name = os.environ.get("PRO_MODEL_NAME", "gemini-1.5-pro-latest")
        flash_model_name = os.environ.get("FLASH_MODEL_NAME", "gemini-1.5-flash-latest")

        self.chat_model = genai.GenerativeModel(
            model_name=flash_model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        self.planner_model = genai.GenerativeModel(model_name=pro_model_name)

        self.tools = {
            "create_file": create_file, "read_file": read_file,
            "update_file": update_file, "git_commit": git_commit,
            "git_status": git_status, "solve_math_problem": solve_math_problem,
        }
        self.executor_model = genai.GenerativeModel(model_name=flash_model_name, tools=list(self.tools.values()))

        # Initialize handlers
        self.chat_handler = ChatModeHandler(progress_manager)
        self.math_handler = MathModeHandler(progress_manager, self.chat_model)
        # Pass _mark_step_complete as a callable to the TaskModeHandler
        self.task_handler = TaskModeHandler(progress_manager, self.planner_model, self.executor_model, self.tools, self._mark_step_complete)

        AGENT_LOGGER.info("Chief Architect Agent (Refactored) initialized successfully.")

    async def _mark_step_complete(self, step_id: int, detail: str = "Completed successfully.") -> dict:
        if self.current_task_context:  # This will be set before execution
            await self.current_task_context.update_step_status(int(step_id), "completed", detail)
            return {"status": "success", "message": f"Step {int(step_id)} marked as complete."}
        return {"status": "error", "reason": "No active task context."}

    async def decide_mode(self, prompt: str) -> str:
        math_keywords = ['calculate', 'solve', 'what is', 'math', 'equation', 'integral', 'derivative', 'algebra',
                         'geometry']
        task_keywords = ['create', 'build', 'update', 'run', 'execute', 'develop', 'implement', 'make', 'write', 'file',
                         'code', 'project']

        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in math_keywords):
            return "math"
        if any(keyword in prompt_lower for keyword in task_keywords):
            return "task"
        if len(prompt.split()) < 6:
            return "chat"
        return "task"

    async def execute_task(self, prompt: str, deep_reasoning: bool = False):
        AGENT_LOGGER.info(f"Executing new task with prompt: '{prompt[:50]}...'")
        self.current_task_context = None  # Ensure no leftover context

        try:
            mode = await self.decide_mode(prompt)
            if mode == "chat":
                await self.chat_handler.handle(prompt)
            elif mode == "math":
                await self.math_handler.handle(prompt)
            else:  # mode == "task"
                self.current_task_context = TaskExecutionContext(prompt, self.progress_manager)
                await self.task_handler.handle(prompt, self.current_task_context, deep_reasoning)
        except Exception as e:
            AGENT_LOGGER.error(f"Task execution failed for prompt '{prompt}': {e}", exc_info=True)
            await self.progress_manager.broadcast("log", f"ERROR: Task failed - {e}")
            await self.progress_manager.broadcast("final_result", f"The agent failed to complete the task. Error: {e}")
        finally:
            AGENT_LOGGER.info(f"Finished task execution for prompt: '{prompt[:50]}'")
