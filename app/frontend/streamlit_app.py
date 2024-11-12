#!/usr/bin/env python3

'''
Talk2BioModels: Interactive BioModel Simulation Tool
'''

import os
import sys
import streamlit as st
sys.path.append('./')
from aiagents4pharma.talk2biomodels.tools.ask_question import AskQuestionTool
from aiagents4pharma.talk2biomodels.tools.simulate_model import SimulateModelTool
from aiagents4pharma.talk2biomodels.tools.plot_figure import PlotImageTool
from aiagents4pharma.talk2biomodels.tools.model_description import ModelDescriptionTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Talk2BioModels", page_icon="🤖", layout="wide")

st.title("🤖 Talk2BioModels: Interactive BioModel Simulation Tool",
        anchor="top",
        help="An interactive tool for running and analyzing biological simulations.")

st.logo(image="./app/frontend/VPE.png", link="https://www.github.com/virtualpatientengine")

uploaded_file = st.file_uploader(
"Upload an XML/SBML file",
accept_multiple_files=False,
type=["xml", "sbml"],
help='''Upload an XML/SBML file to simulate a biological model, \
    and ask questions about the simulation results.'''
)

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

tools = [simulate_model, ask_question, plot_figure, model_description]

# Load the prompt for the main agent
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
# llm = llm.bind_tools(tools)

# Create an agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True,
                               return_intermediate_steps=True)

# React to user input
if prompt := st.chat_input("Say something...", key="user_input"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # A key must added to the session state 
    # to store the object of the last model.
    # This key is used to store the model object
    # and pass it to the next tool.
    if 'last_model_object' not in st.session_state:
        st.session_state.last_model_object = None

    # Create a key 'uploaded_file' to read the uploaded file
    if uploaded_file:
        st.session_state.sbml_file_path = uploaded_file.read().decode("utf-8")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Fetching responses ..."):
            response = agent_executor.invoke({"input": prompt,
                                        "st_session_key": "last_model_object"})
            st.markdown(response["output"])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant",
                                    "content": response["output"]})
