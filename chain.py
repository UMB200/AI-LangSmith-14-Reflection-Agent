import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser, PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schema import RespondToQuestion, ReviseAnswer

llm_chatgpt = ChatOpenAI(model="o4-mini")
parser = JsonOutputToolsParser(return_id=True)
pydantic_parser = PydanticToolsParser(tools=[RespondToQuestion])

actor_prompt = ChatPromptTemplate.from_messages([(
    "system", """You are expert researcher. 
    Current time: {time}
    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvment.
    3. Recommend search queries to research information and improve your answer.""")]).partial(
        time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt = actor_prompt.partial(first_instruction="Provide a detailed ~250 word answer")

first_responder = first_responder_prompt | llm_chatgpt.bind_tools(
    tools=[RespondToQuestion], tool_choice="RespondToQuestion")

revise_instruction = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

revisor = actor_prompt.partial(first_instruction=revise_instruction) | llm_chatgpt.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital")
        
    chain = (first_responder_prompt 
        | llm_chatgpt.bind_tools(tools=[RespondToQuestion], tool_choice="RespondToQuestion")
        | pydantic_parser)

    result = chain.invoke(input={"messages": [human_message]})
    print(result)
