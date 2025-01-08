#!/usr/bin/env python3

'''
Talk2BioModels: Interactive BioModel Simulation Tool
'''

import os
import sys
import random
import uuid
import hmac
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_feedback import streamlit_feedback
sys.path.append('./')
from aiagents4pharma.talk2biomodels.tools.ask_question import AskQuestionTool
from aiagents4pharma.talk2biomodels.tools.simulate_model import SimulateModelTool
from aiagents4pharma.talk2biomodels.tools.model_description import ModelDescriptionTool
from aiagents4pharma.talk2biomodels.tools.search_models import SearchModelsTool
from aiagents4pharma.talk2biomodels.tools.custom_plotter import CustomPlotterTool
from aiagents4pharma.talk2biomodels.tools.fetch_parameters import FetchParametersTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tracers.context import collect_runs
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client

st.set_page_config(page_title="Talk2BioModels", page_icon="🤖", layout="wide")
st.logo(image="./app/frontend/VPE.png", link="https://www.github.com/virtualpatientengine")

def check_login():
    """Returns `True` if the user is logged in."""

    def entered_values():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

        if " " not in st.session_state["user_name"] and '@' not in st.session_state["user_name"]:
            st.session_state["user_name_correct"] = True
        else:
            st.session_state["user_name_correct"] = False
    # Return True if the password and username are validated.
    if st.session_state.get("password_correct", False) and \
        st.session_state.get("user_name_correct", False):
        return True
    # Show input for user name.
    st.text_input(
                "Username",
                key="user_name",
                help="Please enter a name without spaces and @ symbol. \
                    This will be used to personalize the app and for feedback.",
                # on_change=entered_values
                )
    # Show input for password.
    st.text_input(
                "Password",
                type="password",
                key="password",
                help="Please enter the password shared with you.",
                # on_change=entered_values
                )
    st.button("Login", on_click=entered_values)
    if "user_name_correct" in st.session_state:
        if not st.session_state["user_name_correct"]:
            st.error("😕 Please enter a username without spaces and @ symbol")
    if "password_correct" in st.session_state:
        if not st.session_state["password_correct"]:
            st.error("😕 Password incorrect")
    return False

# Check loging if .streamlit/secrets.toml exists
if os.path.exists(".streamlit/secrets.toml"):
    if not check_login():
        st.stop()  # Do not continue if check_login is not True.
else:
    # Set the default user_name as default
    st.session_state.user_name = "default"

# Generate a unique project name for the session
# Set the project name as the user_name + a unique identifier
# This will be used to track the user's session and feedback
if "project_name" not in st.session_state:
    st.session_state.project_name = str(st.session_state.user_name) + '@' + str(uuid.uuid4())

# Set the streamlit session key for the sys bio model
ST_SYS_BIOMODEL_KEY = "last_model_object"

# Define error message
ERROR_MSG = "Sorry, your request could not be \
            processed due to an error. I have logged \
            the error and reported it to the developers. \
            Please try again with a different prompt."

# Define tools and their metadata
simulate_model = SimulateModelTool(st_session_key=ST_SYS_BIOMODEL_KEY)
ask_question = AskQuestionTool(st_session_key=ST_SYS_BIOMODEL_KEY)
with open('./app/frontend/prompts/prompt_ask_question.txt', 'r', encoding='utf-8') as file:
    prompt_content = file.read()
ask_question.metadata = {
    "prompt": prompt_content
}
# plot_figure = PlotImageTool(st_session_key=ST_SYS_BIOMODEL_KEY)
model_description = ModelDescriptionTool(st_session_key=ST_SYS_BIOMODEL_KEY)
with open('./app/frontend/prompts/prompt_model_description.txt', 'r', encoding='utf-8') as file:
    prompt_content = file.read()
model_description.metadata = {
    "prompt": prompt_content
}
search_models = SearchModelsTool()
custom_plotter = CustomPlotterTool(st_session_key=ST_SYS_BIOMODEL_KEY)
fetch_parameters = FetchParametersTool(st_session_key=ST_SYS_BIOMODEL_KEY)

tools = [simulate_model,
        ask_question,
        #  plot_figure,
        custom_plotter,
        fetch_parameters,
        model_description,
        search_models]

# Load the prompt for the main agent
with open('./app/frontend/prompts/prompt_general.txt', 'r', encoding='utf-8') as file:
    prompt_content = file.read()

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_content),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize run_id
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Check if env variable OPENAI_API_KEY exists
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment \
        variable in the terminal where you run the app.")
    st.stop()

# Initialize the OpenAI model
llm = ChatOpenAI(temperature=0.0,
                model="gpt-4o-mini",
                streaming=True,
                api_key=os.getenv("OPENAI_API_KEY"))

# Create an agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True,
                               return_intermediate_steps=True)

def render_plotly(df_simulation_results: pd.DataFrame) -> px.line:
    """
    Function to visualize the dataframe using Plotly.

    Args:
        df: pd.DataFrame: The input dataframe
    """
    df_simulation_results = df_simulation_results.melt(id_vars='Time',
                            var_name='Parameters',
                            value_name='Concentration')
    fig = px.line(df_simulation_results,
                    x='Time',
                    y='Concentration',
                    color='Parameters',
                    title="Concentration of parameters over time",
                    height=500,
                    width=600
            )
    return fig

def get_random_spinner_text():
    """
    Function to get a random spinner text.
    """
    spinner_texts = [
        "Your request is being carefully prepared. one moment, please.",
        "Working on that for you now—thanks for your patience.",
        "Hold tight! I’m getting that ready for you.",
        "I’m on it! Just a moment, please.",
        "Running algorithms... your answer is on its way.",
        "Processing your request. Please hold on...",
        "One moment while I work on that for you...",
        "Fetching the details for you. This won’t take long.",
        "Sit back while I take care of this for you."]
    return random.choice(spinner_texts)

def _submit_feedback(user_response):
    '''
    Function to submit feedback to the developers.
    '''
    client = Client()
    client.create_feedback(
        st.session_state.run_id,
        key="feedback",
        score=1 if user_response['score'] == "👍" else 0,
        comment=user_response['text']
    )
    st.info("Your feedback is on its way to the developers. Thank you!", icon="🚀")
    # return user_response.update({"some metadata": 123})

# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    with st.container(border=True):
        # Title
        st.write("""
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            🤖 Talk2BioModels
            </h3>
            """,
            unsafe_allow_html=True)

        # LLM panel
        llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        llm_option = st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="st_selectbox_llm"
        )

        # Upload files
        uploaded_file = st.file_uploader(
            "Upload an XML/SBML file",
            accept_multiple_files=False,
            type=["xml", "sbml"],
            help='''Upload an XML/SBML file to simulate a biological model, \
                and ask questions about the simulation results.'''
            )

    with st.container(border=False, height=500):
        prompt = st.chat_input("Say something ...", key="st_chat_input")

# Second column
with main_col2:
    # Chat history panel
    with st.container(border=True, height=575):
        st.write("#### 💬 Chat History")

        # Display chat messages
        for count, message in enumerate(st.session_state.messages):
            if message["type"] == "message":
                with st.chat_message(message["content"].role,
                                     avatar="🤖" 
                                     if message["content"].role != 'user'
                                     else "👩🏻‍💻"):
                    st.markdown(message["content"].content)
                    st.empty()
            elif message["type"] == "plotly":
                st.plotly_chart(render_plotly(message["content"]),
                                use_container_width = True,
                                key=f"plotly_{count}")
            elif message["type"] == "dataframe":
                st.dataframe(message["content"],
                            use_container_width = True,
                            key=f"dataframe_{count}")
        if prompt:
            if ST_SYS_BIOMODEL_KEY not in st.session_state:
                st.session_state[ST_SYS_BIOMODEL_KEY] = None

            # Create a key 'uploaded_file' to read the uploaded file
            if uploaded_file:
                st.session_state.sbml_file_path = uploaded_file.read().decode("utf-8")

            # Display user prompt
            prompt_msg = ChatMessage(prompt, role="user")
            st.session_state.messages.append(
                {
                    "type": "message",
                    "content": prompt_msg
                }
            )
            with st.chat_message("user", avatar="👩🏻‍💻"):
                st.markdown(prompt)
                st.empty()

            with st.chat_message("assistant", avatar="🤖"):
                # with st.spinner("Fetching response ..."):
                with st.spinner(get_random_spinner_text()):
                    history = [(m["content"].role, m["content"].content)
                                            for m in st.session_state.messages
                                            if m["type"] == "message"]
                    chat_history = [
                        SystemMessage(content=m[1]) if m[0] == "system" else
                        HumanMessage(content=m[1]) if m[0] == "human" else
                        AIMessage(content=m[1])
                        for m in history
                    ]
                    ERROR_FLAG = False
                    with collect_runs() as cb:
                        # Call the agent
                        try:
                            tracer = LangChainTracer(project_name=st.session_state.project_name)
                            response = agent_executor.invoke({
                                                    "input": prompt,
                                                    "chat_history": chat_history},
                                                    config={"callbacks": [tracer]})
                            # st.markdown(cb.traced_runs[0].id)
                        except Exception as e:
                            ERROR_FLAG = True
                        st.session_state.run_id = cb.traced_runs[0].id

                    # Check if there was an error
                    # If there was an error, display an error message
                    # Otherwise, display the response
                    if ERROR_FLAG:
                        # Add assistant response to chat history
                        assistant_msg = ChatMessage(ERROR_MSG, role="assistant")
                        st.session_state.messages.append({
                                        "type": "error_message",
                                        "content": ERROR_MSG
                                    })
                        # Display the error message
                        st.error(ERROR_MSG, icon="🚨")
                        st.empty()
                        # st.stop()
                    else:
                        # Add assistant response to chat history
                        assistant_msg = ChatMessage(response["output"], role="assistant")
                        st.session_state.messages.append({
                                        "type": "message",
                                        "content": assistant_msg
                                    })
                        # Display the response
                        st.markdown(response["output"])
                        st.empty()

                        # print (response)
                        if "intermediate_steps" in response or len(response["intermediate_steps"]) != 0:
                            for r in response["intermediate_steps"]:
                                if r[0].tool == 'simulate_model':
                                    model_obj = st.session_state[ST_SYS_BIOMODEL_KEY]
                                    df_sim_results = model_obj.simulation_results
                                    # Add data to the chat history
                                    st.session_state.messages.append({
                                                    "type": "dataframe",
                                                    "content": df_sim_results
                                                })
                                    st.dataframe(df_sim_results, use_container_width=True)
                                    # Add the plotly chart to the chat history
                                    st.session_state.messages.append({
                                                    "type": "plotly",
                                                    "content": df_sim_results
                                                })
                                    # Display the plotly chart
                                    st.plotly_chart(render_plotly(df_sim_results),
                                                        use_container_width = True)
                                elif r[0].tool == 'custom_plotter':
                                    model_obj = st.session_state[ST_SYS_BIOMODEL_KEY]
                                    # Prepare df_subset for custom_simulation_results
                                    df_subset = pd.DataFrame()
                                    if len(st.session_state.custom_simulation_results) > 0:
                                        custom_headers = st.session_state.custom_simulation_results
                                        custom_headers = list(custom_headers)
                                        # Add Time column to the custom headers
                                        if 'Time' not in custom_headers:
                                            custom_headers = ['Time'] + custom_headers
                                        # Make df_subset with only the custom headers
                                        df_subset = model_obj.simulation_results[custom_headers]
                                        # Add data to the chat history
                                        st.session_state.messages.append({
                                                        "type": "dataframe",
                                                        "content": df_subset
                                                    })
                                        st.dataframe(df_subset, use_container_width=True)
                                        # Add the plotly chart to the chat history
                                        st.session_state.messages.append({
                                                        "type": "plotly",
                                                        "content": df_subset
                                                    })
                                        # Display the plotly chart
                                        st.plotly_chart(render_plotly(df_subset),
                                                            use_container_width = True)

        # Collect feedback and display the thumbs feedback
        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=_submit_feedback,
                key=f"feedback_{st.session_state.run_id}"
            )
            # print (feedback)
