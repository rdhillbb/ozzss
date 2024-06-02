import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llmobjects import LLMObject

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
output_parser = StrOutputParser()

llmlist = [
    "ChatOpenAI:gpt-4o",
    "ChatAnthropic:claude-1.3",
    "ChatDeepInfra:meta-llama/Meta-Llama-3-70B-Instruct",
    "ChatDeepInfra:mistralai/Mixtral-8x22B-Instruct-v0.1",
    "ChatDeepInfra:google/gemma-1.1-7b-it",
    "ChatDeepInfra:01-ai/Yi-34B-Chat",
    "ChatOllama:llama3:latest",
    "ChatGroq:mixtral-8x7b-32768",
    "ChatGoogleGenerativeAI:gemini-pro",
    "ChatNVIDIA:meta/llama2-70b",
]

for llm in llmlist:
    print()
    print("Platform:", llm)
    llm_instance = LLMObject(llm, temperature=1, max_tokens=4096).llm_instance
    chain2 = prompt | llm_instance | output_parser
    print(chain2.invoke({"topic": "ice cream"}))
    print()
    print("+" * 40)
