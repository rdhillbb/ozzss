import os
from langchain_community.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Make sure to get your API key from DeepInfra. You have to Login and get a new token.
#os.environ["DEEPINFRA_API_TOKEN"] = '<your Deep Infra API token>'

# Create the DeepInfra instance. You can view a list of available parameters in the model page
llm = DeepInfra(model_id="meta-llama/Llama-2-70b-chat-hf",verbose=True)
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1,
    "top_p": 0.9,

}


def example1():
    # run inference
    print(llm.invoke("Who let the dogs out?"))

def example2():
    # run streaming inference
    for chunk in llm.stream("Who let the dogs out?"):
        print(chunk)

def example3():
    # create a prompt template for Question and Answer
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # initiate the LLMChain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # provide a question and run the LLMChain
    question = "Can penguins reach the North pole?"
    print(llm_chain.invoke(question))


# run examples
example1()

