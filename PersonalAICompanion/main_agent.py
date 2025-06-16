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
import base64
from PIL import Image
import io

# Import our modular tools and the new prompt templates
from tools.git_tools import git_commit, git_status
from tools.file_system_tools import create_file, read_file, update_file, get_fs_state_str
from tools.math_tools import solve_math_problem
from tools.physics_tools import solve_physics_problem
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

    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.progress_manager = agent_instance.progress_manager

    @abstractmethod
    async def handle(self, prompt: str, image_data: str | None = None, task_context: TaskExecutionContext | None = None,
                     deep_reasoning: bool = False):
        pass


class ChatModeHandler(BaseModeHandler):
    """Handles simple text-only chat interactions."""

    async def handle(self, prompt: str, image_data: str | None = None, task_context: TaskExecutionContext | None = None,
                     deep_reasoning: bool = False):
        await self.progress_manager.broadcast("log", "Engaging in chat mode...")
        chat_prompt = f"You are a helpful AI assistant. A user said: '{prompt}'. Respond in a friendly, direct manner."

        response_stream = await self.agent.chat_model.generate_content_async(chat_prompt, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
            except (ValueError, IndexError):
                AGENT_LOGGER.debug("Received an empty stream chunk.")
                pass
        await self.progress_manager.broadcast("final_result", "Chat interaction complete.")


class VisionModeHandler(BaseModeHandler):
    """Handles general vision-related questions."""

    async def handle(self, prompt: str, image_data: str | None = None, task_context: TaskExecutionContext | None = None,
                     deep_reasoning: bool = False):
        await self.progress_manager.broadcast("log", "Engaging in Vision Mode...")

        system_instruction = "You are an expert at analyzing images. Your task is to respond to the user's request based on the provided image."
        content = [system_instruction]

        if prompt:
            content.append(prompt)

        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                content.append(img)
                await self.progress_manager.broadcast("log", "Image successfully processed for analysis.")
            except Exception as e:
                AGENT_LOGGER.error(f"Could not process image data for vision mode: {e}")
                await self.progress_manager.broadcast("log", "Error: The attached image could not be processed.")
                return
        else:
            await self.agent.chat_handler.handle(
                "It seems you wanted me to analyze an image, but I didn't receive one. Please try again.")
            return

        response_stream = await self.agent.vision_model.generate_content_async(content, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
            except (ValueError, IndexError):
                pass
        await self.progress_manager.broadcast("final_result", "Vision analysis complete.")


class PhysicsModeHandler(BaseModeHandler):
    """Handles physics problems, including those from images."""

    async def handle(self, prompt: str, image_data: str | None = None, task_context: TaskExecutionContext | None = None,
                     deep_reasoning: bool = False):
        await self.progress_manager.broadcast("log", "Engaging in Physics Mode (with Vision)...")

        content = [prompt] if prompt else ["Analyze the attached image."]
        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                content.append(img)
                await self.progress_manager.broadcast("log", "Image successfully processed for physics problem.")
            except Exception as e:
                AGENT_LOGGER.error(f"Could not process image data: {e}")
                await self.progress_manager.broadcast("log", "Error: The attached image could not be processed.")
                return

        solution_prompt = (
            "You are an expert physics tutor. A user has a physics problem, potentially including an image. "
            "Your first task is to transcribe the full problem, including all text and values from the image if one is provided. "
            "Then, provide a complete, step-by-step solution. "
            "Format your entire response as a single JSON object with a key 'solution'. "
            "The value of 'solution' must be an array of objects, where each object has a 'type' ('text' or 'latex') and 'content'. "
            "Use 'text' for explanations and 'latex' for all mathematical formulas, equations, and variables. "
            "Ensure the LaTeX is correctly formatted for rendering."
        )
        content.insert(0, solution_prompt)

        response = await self.agent.json_vision_model.generate_content_async(content)

        try:
            solution_data = json.loads(response.text)
            for step in solution_data.get("solution", []):
                message_type = "chat_chunk" if step.get("type") == "text" else "latex_canvas"
                await self.progress_manager.broadcast(message_type, step.get("content", ""))
        except (json.JSONDecodeError, AttributeError) as e:
            AGENT_LOGGER.error(f"Failed to parse physics solution JSON: {e} | Response: {response.text}")
            await self.progress_manager.broadcast("log",
                                                  "I had trouble formatting the physics solution. Please try again.")

        await self.progress_manager.broadcast("final_result", "Physics problem solved.")


class TaskModeHandler(BaseModeHandler):
    """Handles complex multi-step tasks involving planning and tool execution."""

    async def _perform_planning(self, task_context: TaskExecutionContext, deep_reasoning: bool,
                                image_data: str | None = None):
        await self.progress_manager.broadcast("log",
                                              f"Phase 1: Planning (Mode: {'Deep Reasoning' if deep_reasoning else 'Standard'})...")

        planning_content = []
        git_status_output = git_status().get('output', 'Not available.')
        project_state = f"--- File System ---\n{get_fs_state_str()}\n--- Git Status ---\n{git_status_output}"

        prompt_template = get_deep_reasoning_prompt if deep_reasoning else get_standard_planning_prompt
        planning_content.append(prompt_template(task_context.original_prompt, project_state))

        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                planning_content.append(img)
                await self.progress_manager.broadcast("log", "Planner is considering the attached image.")
            except Exception as e:
                AGENT_LOGGER.error(f"Could not process image for planner: {e}")

        response_stream = await self.agent.planner_model.generate_content_async(planning_content, stream=True)
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
            await task_context.set_plan(json.loads(json_match.group(1)))
        else:
            raise ValueError("LLM did not return a valid JSON plan from its reasoning stream.")

    async def _perform_execution(self, task_context: TaskExecutionContext):
        await self.progress_manager.broadcast("log", "Phase 2: Execution...")
        chat_session = self.agent.executor_model.start_chat()
        for step in task_context.plan:
            if step.get('status') == 'completed': continue
            await task_context.update_step_status(step['id'], 'in_progress')
            git_status_output = git_status().get('output', 'Not available.')
            project_state = f"--- File System ---\n{get_fs_state_str()}\n--- Git Status ---\n{git_status_output}"
            next_message = f"Executing Step {step['id']}: {step['task']}\nUse tools to complete this. Call 'mark_step_complete' when done.\nState:\n{project_state}"
            step_completed = False
            for _ in range(7):
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
                    if tool_name in self.agent.tools:
                        await task_context.add_to_scratchpad('TOOL_REQUEST', f"Calling {tool_name} with args {args}")
                        if tool_name == 'mark_step_complete':
                            result = await self.agent.tools[tool_name](**args)
                            step_completed = True
                        else:
                            result = self.agent.tools[tool_name](**args)
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
        response_stream = await self.agent.chat_model.generate_content_async(summary_prompt, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
            except (ValueError, IndexError):
                AGENT_LOGGER.debug("Received an empty stream chunk during final response.")
                pass

    async def handle(self, prompt: str, image_data: str | None = None, task_context: TaskExecutionContext | None = None,
                     deep_reasoning: bool = False):
        if not task_context:
            raise ValueError("TaskModeHandler requires a TaskExecutionContext.")
        await self._perform_planning(task_context, deep_reasoning, image_data)
        await self._perform_execution(task_context)
        await self._generate_final_response(task_context)
        await self.progress_manager.broadcast("final_result", "Task completed successfully.")


class ChiefArchitectAgent:
    def __init__(self, progress_manager: ProgressManager):
        AGENT_LOGGER.info("Initializing ChiefArchitectAgent instance...")
        load_dotenv()
        api_key = os.getenv("API_KEY")
        if not api_key: raise ValueError("API_KEY not found.")
        genai.configure(api_key=api_key)

        self.progress_manager = progress_manager

        pro_model_name = "gemini-1.5-pro-latest"
        flash_model_name = "gemini-1.5-flash-latest"

        # Models
        self.chat_model = genai.GenerativeModel(model_name=flash_model_name)
        self.vision_model = genai.GenerativeModel(model_name=pro_model_name)
        self.json_vision_model = genai.GenerativeModel(
            model_name=pro_model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        self.planner_model = genai.GenerativeModel(model_name=pro_model_name)

        # Tools and Executor
        self.tools = {
            "create_file": create_file, "read_file": read_file,
            "update_file": update_file, "git_commit": git_commit,
            "git_status": git_status, "solve_math_problem": solve_math_problem,
            "solve_physics_problem": solve_physics_problem,
            "mark_step_complete": self._mark_step_complete
        }
        self.executor_model = genai.GenerativeModel(model_name=flash_model_name, tools=list(self.tools.values()))

        # Mode Handlers
        self.chat_handler = ChatModeHandler(self)
        self.physics_handler = PhysicsModeHandler(self)
        self.vision_handler = VisionModeHandler(self)
        self.task_handler = TaskModeHandler(self)

        AGENT_LOGGER.info("Chief Architect Agent (V18 - Final Vision Fix) initialized successfully.")

    async def _mark_step_complete(self, step_id: int, detail: str = "Completed successfully.") -> dict:
        if self.current_task_context:
            await self.current_task_context.update_step_status(int(step_id), "completed", detail)
            return {"status": "success", "message": f"Step {int(step_id)} marked as complete."}
        return {"status": "error", "reason": "No active task context."}

    async def decide_mode(self, prompt: str, image_data: str | None = None) -> str:
        physics_keywords = ['physics', 'force', 'mass', 'acceleration', 'velocity', 'position', 'momentum', 'energy',
                            'integral', 'derivative', 'kinematics', 'solve', 'calculate']
        task_keywords = ['create', 'build', 'update', 'run', 'execute', 'develop', 'implement', 'make', 'write', 'file',
                         'code', 'project']
        # --- MODIFICATION START ---
        # Add vision keywords to detect image-related prompts from the text itself.
        vision_keywords = ['image', 'picture', 'see', 'look', 'analyze', 'describe', 'transcribe', 'attached']
        prompt_lower = prompt.lower() if prompt else ""

        # If image_data is present OR if the prompt contains vision-related keywords,
        # we should use a vision-capable handler.
        has_vision_keyword = any(keyword in prompt_lower for keyword in vision_keywords)

        if image_data or has_vision_keyword:
            if any(keyword in prompt_lower for keyword in physics_keywords):
                return "physics"
            # This now correctly routes "transcribe this image" to vision mode
            # based on the keywords in the prompt.
            return "vision"
        # --- MODIFICATION END ---

        # Text-only routing
        if any(keyword in prompt_lower for keyword in physics_keywords): return "physics"
        if any(keyword in prompt_lower for keyword in task_keywords): return "task"
        if len(prompt.split()) < 6: return "chat"
        return "task"

    async def execute_task(self, prompt: str, image_data: str | None = None, deep_reasoning: bool = False):
        AGENT_LOGGER.info(f"Executing new task with prompt: '{prompt[:50]}...'")
        self.current_task_context = None

        try:
            mode = await self.decide_mode(prompt, image_data)

            if mode == "chat":
                await self.chat_handler.handle(prompt, image_data, deep_reasoning=deep_reasoning)
            elif mode == "physics":
                await self.physics_handler.handle(prompt, image_data, deep_reasoning=deep_reasoning)
            elif mode == "vision":
                await self.vision_handler.handle(prompt, image_data, deep_reasoning=deep_reasoning)
            else:  # mode == "task"
                self.current_task_context = TaskExecutionContext(prompt, self.progress_manager)
                await self.task_handler.handle(prompt, image_data, task_context=self.current_task_context,
                                               deep_reasoning=deep_reasoning)
        except Exception as e:
            AGENT_LOGGER.error(f"Task execution failed for prompt '{prompt}': {e}", exc_info=True)
            await self.progress_manager.broadcast("log", f"ERROR: Task failed - {e}")
            await self.progress_manager.broadcast("final_result", f"The agent failed to complete the task. Error: {e}")
        finally:
            AGENT_LOGGER.info(f"Finished task execution for prompt: '{prompt[:50]}'")
