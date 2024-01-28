import os, openai

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish

from langchain_community.tools import YouTubeSearchTool

from langgraph.graph import END, Graph
import streamlit as st

def initialize(openai_api_key):

    if not openai_api_key:
        print("Can't initialize. OpenAI key is missing")
        return

    #openai.api_key=key
    os.environ["OPENAI_API_KEY"]=openai_api_key
   
    # Get the prompt to use (default
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, temperature=0.2)

    from langchain_community.tools import YouTubeSearchTool
    tools = [YouTubeSearchTool()]

    agent_runnable = create_openai_functions_agent(llm, tools, prompt)

    # Define the agent
    agent = RunnablePassthrough.assign(
        agent_outcome = agent_runnable
    )

    # Define the function to execute tools
    def execute_tools(data):
        agent_action = data.pop('agent_outcome')
        tool_to_use = {t.name: t for t in tools}[agent_action.tool]
        observation = tool_to_use.invoke(agent_action.tool_input)
        data['intermediate_steps'].append((agent_action, observation))
        print(data)
        return data

    # Define logic that will be used to determine which conditional edge to go down
    def should_continue(data):
        if isinstance(data['agent_outcome'], AgentFinish):
            return "exit"
        else:
            return "continue"
        
    workflow = Graph()

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools)

    # Set the entrypoint as `agent`
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "exit": END
        }
    )
    workflow.add_edge('tools', 'agent')

    # This compiles it into a LangChain Runnable,
    # meaning we can use it as you would any other runnable
    chain = workflow.compile()

    return chain


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    
st.title("💬 Search Youtube")
st.caption("🚀 A streamlit agent powered by OpenAI LLM to search Youtube")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

chain = initialize(openai_api_key)

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
 
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
   
    result = chain.invoke({"input": prompt, "intermediate_steps": []})
    msg = result['agent_outcome'].return_values["output"]
    
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)