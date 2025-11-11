from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="deepseek-chat")

SYSTEM_PROMPT = """You are a focused research assistant who uses available tools to gather
up-to-date information. Verify facts, cite concrete sources, and only rely on a tool's
output if it supports the final answer."""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(AgentResponse),
)


def main():
    user_prompt = (
        "search for 3 job posting for an ai engineer using langchain in the bay area "
        "on linkedin and list their details"
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        }
    )

    structured_response = result.get("structured_response")
    if structured_response is None:
        raise ValueError("Agent did not return a structured response.")

    print(structured_response)


if __name__ == "__main__":
    main()
