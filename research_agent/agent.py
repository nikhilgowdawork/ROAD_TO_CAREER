
import json
from typing import Any

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from research_agent.llm import get_llm
from research_agent.tools.websearch import search_web
from research_agent.prompts import INDUSTRY_RESEARCH_PROMPT


MAX_ITERATIONS = 6

SYSTEM_PROMPT = INDUSTRY_RESEARCH_PROMPT

def compact_search_result(result: dict) -> dict:
    """Keep search evidence concise and usable."""
    if "results" not in result:
        return result

    return {
        "query": result.get("query", ""),
        "result_count": result.get("result_count", 0),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:1500],
                "score": item.get("score", 0),
            }
            for item in result["results"][:3]
        ],
    }


def run_research(query: str) -> str:
    """Execute an iterative, tool-using industry research workflow."""

    if not query or not query.strip():
        raise ValueError("Research query cannot be empty.")

    llm = get_llm()

    tools = [search_web]
    tools_by_name = {tool.name: tool for tool in tools}

    llm_with_tools = llm.bind_tools(
        tools,
        tool_choice="auto",
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query.strip()),
    ]

    research_evidence: list[dict[str, Any]] = []

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Research iteration {iteration + 1} ---")

        # Preserve the original request and recent tool interactions.
        if len(messages) > 7:
            messages = [
                messages[0],
                messages[1],
                *messages[-5:],
            ]

        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        # No tool calls means the model has produced its final answer.
        if not ai_response.tool_calls:
            return ai_response.content or "The model returned an empty response."

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            print(f"Tool requested: {tool_name}")
            print(f"Arguments: {tool_args}")

            selected_tool = tools_by_name.get(tool_name)

            if selected_tool is None:
                result = {
                    "error": f"Unknown tool: {tool_name}",
                }
            else:
                try:
                    result = selected_tool.invoke(tool_args)

                    if isinstance(result, dict):
                        result = compact_search_result(result)
                        research_evidence.append(result)

                except Exception as exc:
                    result = {
                        "error": (
                            f"{type(exc).__name__}: {str(exc)}"
                        )
                    }

            result_text = json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )

            print(f"Tool result length: {len(result_text)} characters")

            messages.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call_id,
                )
            )

    # Iteration limit reached: synthesize using a fresh conversation.
    # This avoids carrying pending tool-call history into final synthesis.
    print("\nResearch iteration limit reached. Synthesizing findings...")

    final_messages = [
        SystemMessage(
            content=(
                "Produce the final research report using only the "
                "original request and collected evidence below. "
                "Do not call tools or invent facts. Include source URLs "
                "where available, identify evidence gaps, and explain "
                "important uncertainties."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "research_question": query,
                    "collected_evidence": research_evidence,
                },
                ensure_ascii=False,
                default=str,
            )
        ),
    ]

    final_response = llm.invoke(final_messages)

    return final_response.content or "The model returned an empty response."
