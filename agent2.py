import os
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] # HumanMessage and AIMessage are two datatypes in langchain package

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def process(state: AgentState) -> AgentState:
    """this node will solve the request you input"""
    try:
        response = llm.invoke(state['messages'])
        state['messages'].append(AIMessage(content=response.content))
        print(f'\nAI: {response.content}')
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        state['messages'].append(AIMessage(content=error_msg))
        print(f'\nError: {error_msg}')

    return state

graph = StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()

conversation_history = []

user_input = input('Enter: ')
while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({'messages': conversation_history})

    conversation_history = result['messages']

    # Save conversation after each turn
    with open("logging.txt", "w", encoding="utf-8") as file:
        file.write("Your Conversation Log:\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f'You: {message.content}\n')
            elif isinstance(message, AIMessage):
                file.write(f'AI: {message.content}\n\n')
        file.write('End Of Conversation')

    user_input = input('Enter: ')

print("conversation saved to logging.txt")