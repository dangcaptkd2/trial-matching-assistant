"""Workflow service for invoking LangGraph workflow with streaming and tracing."""

from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional

from src.api.models.schemas import ExecutionStep, ExecutionTrace, WorkflowResponse
from src.core.graph import create_workflow

# Create workflow instance (singleton)
_workflow_app = None


def get_workflow_app():
    """Get or create the workflow app instance."""
    global _workflow_app
    if _workflow_app is None:
        _workflow_app = create_workflow()
    return _workflow_app


class WorkflowService:
    """Service to invoke and manage LangGraph workflow execution."""

    def __init__(self):
        """Initialize workflow service."""
        self.app = get_workflow_app()

    async def invoke_workflow(
        self,
        user_input: str,
        thread_id: str,
        top_k: int = 10,
        stream: bool = False,
    ) -> AsyncIterator[Dict]:
        """
        Invoke the LangGraph workflow with streaming support.

        Args:
            user_input: User's input message
            thread_id: Thread ID for conversation continuity
            top_k: Number of search results to retrieve
            stream: Whether to stream execution steps

        Yields:
            Dict with execution events and final result
        """
        # Prepare initial state
        initial_state = {
            "user_input": user_input,
            "messages": [],
            "top_k": top_k,
        }

        # Prepare config with thread_id for checkpointing
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "workflow": "clinical_trial_assistant",
                "top_k": top_k,
            },
            "tags": ["clinical-trials", "chainlit"],
        }

        if stream:
            # Stream execution steps
            async for event in self._stream_workflow(initial_state, config):
                yield event
        else:
            # Single invocation
            result = await self.app.ainvoke(initial_state, config=config)
            yield {
                "type": "result",
                "data": result,
            }

    async def _stream_workflow(
        self, initial_state: Dict, config: Dict
    ) -> AsyncIterator[Dict]:
        """
        Stream workflow execution steps.

        Args:
            initial_state: Initial workflow state
            config: Workflow configuration

        Yields:
            Execution events and final result
        """
        execution_steps: List[ExecutionStep] = []
        execution_path: List[str] = []
        start_time = datetime.now()
        step_times: Dict[str, datetime] = {}
        final_result = {}

        async for event in self.app.astream(initial_state, config=config):
            # Process each event from the stream
            for node_name, node_output in event.items():
                # Track execution path
                if node_name not in execution_path:
                    execution_path.append(node_name)
                    step_times[node_name] = datetime.now()

                # Merge node output into final result
                if isinstance(node_output, dict):
                    final_result.update(node_output)

                # Create execution step for this node
                step_start = step_times.get(node_name, datetime.now())

                # Check if we already have a step for this node
                existing_step = next(
                    (s for s in execution_steps if s.node_name == node_name), None
                )

                if not existing_step:
                    # Create new step
                    step = ExecutionStep(
                        node_name=node_name,
                        status="started",
                        start_time=step_start,
                        input_summary=self._summarize_state(node_output, "input"),
                    )
                    execution_steps.append(step)
                else:
                    # Update existing step
                    existing_step.status = "started"
                    if not existing_step.input_summary:
                        existing_step.input_summary = self._summarize_state(
                            node_output, "input"
                        )

                # Yield step start event
                yield {
                    "type": "step_start",
                    "node": node_name,
                    "data": node_output,
                }

        # Finalize all steps
        end_time = datetime.now()
        for step in execution_steps:
            if step.start_time:
                step.end_time = end_time
                duration = (step.end_time - step.start_time).total_seconds() * 1000
                step.duration_ms = duration
            step.status = "completed"
            if not step.output_summary:
                step.output_summary = "Completed"

        # Create execution trace
        total_duration = (datetime.now() - start_time).total_seconds() * 1000
        execution_trace = ExecutionTrace(
            steps=execution_steps,
            total_duration_ms=total_duration,
            execution_path=execution_path,
        )

        # Format response
        response_text = self._extract_response(final_result)
        response_type = self._determine_response_type(final_result)

        workflow_response = WorkflowResponse(
            response=response_text,
            response_type=response_type,
            execution_trace=execution_trace,
            metadata={"thread_id": config["configurable"]["thread_id"]},
        )

        yield {
            "type": "result",
            "data": final_result,
            "trace": execution_trace,
            "response": workflow_response,
        }

    def _summarize_state(self, state: Dict, context: str = "") -> str:
        """Create a summary of state for display."""
        if isinstance(state, dict):
            keys = list(state.keys())[:5]
            return f"State keys: {', '.join(keys)}"
        return str(state)[:100] if state else ""

    def _extract_response(self, result: Dict) -> str:
        """Extract the final response text from workflow result."""
        if result.get("chitchat_response"):
            return result.get("chitchat_response", "")
        elif result.get("final_answer"):
            return result.get("final_answer", "")
        else:
            return "No response generated."

    def _determine_response_type(self, result: Dict) -> str:
        """Determine the type of response."""
        if result.get("chitchat_response"):
            return "chitchat"
        elif result.get("trial_lookup_results"):
            return "trial_lookup"
        elif result.get("final_answer"):
            return "trial_search"
        return "unknown"

    async def get_conversation_history(self, thread_id: str) -> Optional[Dict]:
        """
        Retrieve conversation history from checkpointer.

        Args:
            thread_id: Thread identifier

        Returns:
            Conversation state with messages or None
        """
        try:
            # Get the latest checkpoint using get_state
            config = {"configurable": {"thread_id": thread_id}}
            # Use the app's get_state method to retrieve checkpoint
            state = self.app.get_state(config)
            if state and state.values:
                return state.values
            return None
        except Exception:
            return None


# Global workflow service instance
workflow_service = WorkflowService()
