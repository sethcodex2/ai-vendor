from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from .database import Query


class SearchInput(BaseModel):
    category: str = Field(description="should be a product category")

class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "useful for when you need to answer questions that requires displaying a given type of product."
    args_schema = SearchInput

    def _run(
        self, query: str, run_manager = None
    ) -> str:
        session = Query()
        products = session.get_products(query)
        return products
        

    async def _arun(
        self, query: str, run_manager = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")




