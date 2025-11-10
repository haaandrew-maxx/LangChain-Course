from ast import parse
from calendar import c

from dotenv import load_dotenv
import prompt

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse
tools = [TavilySearch()]
# llm = ChatOpenAI(model="gpt-4")
llm = ChatOpenAI(model="deepseek-chat")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
# ).partial(format_instructions=output_parser.get_format_instructions())
).partial(format_instructions="")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))

# chain = agent_executor | extract_output | parse_output
chain = agent_executor | extract_output | structured_llm
def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job posting for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
