from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """Search the internet for information."""
    print(f"Searching for: {query}")
    return tavily.search(query=query)


llm = ChatOllama(model="qwen3:8b", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-learning!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job postings for an AI engineer using langchain in Chennai area and list their details"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
