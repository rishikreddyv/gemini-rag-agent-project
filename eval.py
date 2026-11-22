import os
from src.agent import build_agent

TEST_CASES = [
    {
        "name": "resume_citation",
        "prompt": "Summarize the candidate's key skills and projects from the uploaded document with page citations.",
        "must_contain_any": ["Page", "Resume.pdf"],
    },
    {
        "name": "calculator_tool",
        "prompt": "What is (25 * 4) + 18?",
        "must_contain_any": ["118"],
    },
    {
        "name": "missing_document_fact",
        "prompt": "According to the uploaded resume only, what was the candidate's score on the 2029 Mars marathon?",
        "must_contain_any": ["not found", "not mentioned", "does not contain", "no mention"],
    },
]


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n".join(parts)
    return str(content)


def get_langfuse_callback():
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        os.environ.setdefault("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    return None


def run_eval():
    agent = build_agent()
    handler = get_langfuse_callback()
    config = {"callbacks": [handler]} if handler else {}

    passed = 0
    try:
        for case in TEST_CASES:
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": case["prompt"]}]},
                    config=config,
                )
                answer = _extract_text(result["messages"][-1].content)
                matched = any(token.lower() in answer.lower() for token in case["must_contain_any"])
                status = "PASS" if matched else "CHECK"
                if matched:
                    passed += 1
                print(f"[{status}] {case['name']}")
                print(f"  Response: {answer[:220]}{'...' if len(answer) > 220 else ''}\n")
            except Exception as exc:
                print(f"[ERROR] {case['name']}: {exc}\n")
    finally:
        if handler:
            from langfuse import get_client
            get_client().flush()

    print(f"Passed {passed}/{len(TEST_CASES)} checks.")


if __name__ == "__main__":
    run_eval()
