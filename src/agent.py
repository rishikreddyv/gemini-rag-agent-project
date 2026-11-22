import ast
import operator
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.retriever import HybridRerankRetriever

_retriever = HybridRerankRetriever()
_ddg = DuckDuckGoSearchRun()

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_math_node(node.operand))
    raise ValueError("Unsupported math expression.")


@tool
def search_documents(query: str) -> str:
    """Search the uploaded PDF documents for relevant facts, skills, projects, or details. Always include page and file citations in your answer."""
    docs = _retriever.retrieve(query)
    return _retriever.format_with_citations(docs)


@tool
def calculator(expression: str) -> str:
    """Use this for basic arithmetic. Input must be a plain math expression like '(25 * 4) + 10'."""
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_math_node(parsed)
        return str(int(result)) if result.is_integer() else str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def web_search(query: str) -> str:
    """Use this to answer questions about current events or facts not found in the uploaded documents."""
    return _ddg.run(query)


def build_agent():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        api_key=api_key,
    )
    tools = [search_documents, calculator, web_search]
    system_prompt = (
        "You are a helpful AI research assistant. "
        "When answering from uploaded documents, cite the exact [file | Page X] source from the tool output. "
        "If the answer is not in the documents and not a math/current-event query, say clearly that it is not found in the provided document rather than guessing."
    )
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)
