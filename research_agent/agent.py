
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.llm import get_llm
from research_agent.tools.websearch import search_web
from research_agent.prompts import INDUSTRY_RESEARCH_PROMPT


MAX_ITERATIONS = 6
MAX_EVIDENCE_ITEMS = 9
MAX_CONTENT_LENGTH = 700
MAX_MODEL_RETRIES = 2

SYSTEM_PROMPT = INDUSTRY_RESEARCH_PROMPT


def compact_search_result(result: dict) -> dict:
    """Keep only concise, useful search evidence."""

    if "results" not in result:
        return result

    results = sorted(
        result["results"],
        key=lambda item: item.get("score", 0),
        reverse=True,
    )

    return {
        "query": result.get("query", ""),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:MAX_CONTENT_LENGTH],
                "score": item.get("score", 0),
            }
            for item in results[:3]
        ],
    }


def build_research_context(query: str, evidence: list) -> list:
    """Build a bounded context without old tool-call history."""

    compact_evidence = evidence[-MAX_EVIDENCE_ITEMS:]

    return [
        SystemMessage(
            content=(
                SYSTEM_PROMPT
                + "\n\nWhen using search_web, provide exactly one "
                "string argument named query. Do not provide any "
                "other arguments. Make one focused search at a time."
            )
        ),
        HumanMessage(
            content=(
                f"Research question: {query}\n\n"
                f"Evidence collected:\n"
                f"{json.dumps(compact_evidence, ensure_ascii=False)}\n\n"
                "Either request a focused search to fill an important "
                "evidence gap, or produce the final report if the "
                "evidence is sufficient."
            )
        ),
    ]


def invoke_with_retry(llm_with_tools, messages):
    """Retry malformed model tool calls a limited number of times."""

    for attempt in range(MAX_MODEL_RETRIES + 1):
        try:
            return llm_with_tools.invoke(messages)

        except Exception as exc:
            error_text = str(exc).lower()

            malformed_call = (
                "failed to parse tool call arguments" in error_text
                or "tool_use_failed" in error_text
                or "failed to render tokens" in error_text
            )

            if not malformed_call or attempt == MAX_MODEL_RETRIES:
                raise

            print(
                f"Malformed tool call. Retrying "
                f"({attempt + 1}/{MAX_MODEL_RETRIES})..."
            )
            time.sleep(1)


def synthesize_report(llm, query: str, evidence: list) -> str:
    """Generate a final report without binding any tools."""

    final_messages = [
        SystemMessage(
            content=(
                "Write a structured industry research report using "
                "the supplied evidence. Do not call tools or invent "
                "facts. Include source URLs, distinguish supported "
                "findings from uncertainty, and identify evidence gaps."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "research_question": query,
                    "collected_evidence": evidence,
                },
                ensure_ascii=False,
                default=str,
            )
        ),
    ]

    response = llm.invoke(final_messages)
    return response.content or "The model returned an empty report."


def run_research(query: str) -> str:
    """Run iterative web research with bounded evidence and retries."""

    if not query or not query.strip():
        raise ValueError("Research query cannot be empty.")

    query = query.strip()
    llm = get_llm()

    tools = [search_web]
    tools_by_name = {tool.name: tool for tool in tools}

    llm_with_tools = llm.bind_tools(
        tools,
        tool_choice="auto",
    )

    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Research iteration {iteration + 1} ---")

        messages = build_research_context(query, evidence)

        try:
            ai_response = invoke_with_retry(
                llm_with_tools,
                messages,
            )
        except Exception as exc:
            print(f"Model call failed: {type(exc).__name__}: {exc}")

            if evidence:
                print("Generating report from collected evidence.")
                return synthesize_report(llm, query, evidence)

            raise RuntimeError(
                "The model failed before collecting research evidence."
            ) from exc

        if not ai_response.tool_calls:
            return ai_response.content or synthesize_report(
                llm, query, evidence
            )

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args") or {}

            print(f"Tool requested: {tool_name}")
            print(f"Arguments: {tool_args}")

            selected_tool = tools_by_name.get(tool_name)

            if selected_tool is None:
                print(f"Skipping unknown tool: {tool_name}")
                continue

            # Validate and pass only the supported argument.
            search_query = tool_args.get("query")

            if not isinstance(search_query, str) or not search_query.strip():
                print("Skipping invalid search arguments.")
                continue

            try:
                # The tool schema now needs only the query argument.
                result = selected_tool.invoke(
                    {"query": search_query.strip()}
                )

                if not isinstance(result, dict):
                    print("Search tool returned an unexpected result.")
                    continue

                if result.get("error"):
                    print(f"Search failed: {result['error']}")
                    continue

                compact_result = compact_search_result(result)

                for item in compact_result.get("results", []):
                    url = item.get("url", "").strip()

                    if url and url in seen_urls:
                        continue

                    if url:
                        seen_urls.add(url)

                    evidence.append({
                        "query": compact_result.get("query", ""),
                        **item,
                    })

                # Keep the evidence sent to subsequent model calls bounded.
                evidence = evidence[-MAX_EVIDENCE_ITEMS:]

                print(
                    f"Collected {len(compact_result.get('results', []))} "
                    f"search results; retaining {len(evidence)} evidence items."
                )

            except Exception as exc:
                print(f"Tool execution failed: {type(exc).__name__}: {exc}")

    print("\nResearch iteration limit reached. Synthesizing report...")
    return synthesize_report(llm, query, evidence)
