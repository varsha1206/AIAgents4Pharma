#!/usr/bin/env python3

'''
Talk2BioModels: Interactive BioModel Simulation Tool
'''

import os
import sys
import streamlit as st
sys.path.append('./')
from agents.talk2biomodels.tools.ask_question import AskQuestionTool
from agents.talk2biomodels.tools.simulate_model import SimulateModelTool
from agents.talk2biomodels.tools.plot_figure import PlotImageTool
from agents.talk2biomodels.tools.model_description import ModelDescriptionTool
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Talk2BioModels", page_icon="🤖", layout="wide")

st.title("🤖 Talk2BioModels: Interactive BioModel Simulation Tool",
        anchor="top",
        help="An interactive tool for running and analyzing biological simulations.")

st.logo(image="./app/frontend/VPE.png", link="https://www.github.com/virtualpatientengine")

# Define tools and their metadata
simulate_model = SimulateModelTool()
ask_question = AskQuestionTool()
with open('./app/frontend/prompts/prompt_ask_question.txt', 'r', encoding='utf-8') as file:
    prompt_content = file.read()
ask_question.metadata = {
    "prompt": prompt_content
}
plot_figure = PlotImageTool()
model_description = ModelDescriptionTool()
with open('./app/frontend/prompts/prompt_model_description.txt', 'r', encoding='utf-8') as file:
    prompt_content = file.read()
model_description.metadata = {
    "prompt": prompt_content
}

@tool
def check_if_results_exist(st_session_key: str) -> bool:
    """
    Check if the simulation results exist.
    """
    if st_session_key not in st.session_state:
        return False
    return True

tools = [check_if_results_exist,
        simulate_model,
        ask_question,
        plot_figure,
        model_description]

# Load the general prompt
with open('./app/frontend/prompts/prompt_general.txt', 'r', encoding='utf-8') as file:
    prompt_content = file.read()

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_content),
        ("human", "{input} {st_session_key}"),
        ("placeholder", "{agent_scratchpad}"),
])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize the OpenAI model
llm = ChatOpenAI(temperature=0.0,
                 model="gpt-3.5-turbo",
                 api_key=os.getenv("OPENAI_API_KEY"))
llm = llm.bind_tools(tools)

# Create an agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
def generate_response(input_text):
    """
    Generate a response to the user input by calling the agent executor.
    """
    answer = agent_executor.invoke({"input":input_text,
                                    "st_session_key": "last_model_object"})
    return answer

# React to user input
if prompt := st.chat_input("Say something...", key="user_input"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if 'last_model_object' not in st.session_state:
        st.session_state.last_model_object = None

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response["output"])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant",
                                      "content": response["output"]})
