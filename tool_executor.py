from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from schema import RespondToQuestion, ReviseAnswer

tavily_search = TavilySearch(max_result=5)

def run_query(search_queries: list[str], **kwargs):
    """ Run the generated queries"""
    return tavily_search.batch([{"query": query} for query in search_queries])

tool_executor = ToolNode([
    StructuredTool.from_function(run_query, name=RespondToQuestion.__name__),
    StructuredTool.from_function(run_query, name=ReviseAnswer.__name__)])