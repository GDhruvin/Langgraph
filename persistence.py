from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver


# -----------------------------
# 1. Define the Chat State
# -----------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# 2. Initialize LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# -----------------------------
# 3. Define chat node
# -----------------------------
def chat_node(state: ChatState):
    messages = state["messages"]        # take conversation history
    response = llm.invoke(messages)     # send to LLM
    return {"messages": [response]}     # add response back to state


# -----------------------------
# 4. Build the graph
# -----------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# -----------------------------
# 5. Setup persistence
# -----------------------------
memory = SqliteSaver.from_conn_string(":memory:")   # Or "chatbot_state.db"
chatbot = graph.compile(checkpointer=memory)


# -----------------------------
# 6. Run chatbot with persistence
# -----------------------------
thread_id = "user123"  # conversation ID (like session ID)


def run_chatbot(user_input: str):
    """Send message to chatbot and print conversation history"""
    events = chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Print the AI response from this run
    for event in events:
        if "chat_node" in event:
            ai_message = event["chat_node"]["messages"][-1]
            print(f"\n🤖: {ai_message.content}")

    # Retrieve full conversation history (to show persistence)
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    print("\n📜 Full conversation so far:")
    for msg in state.values["messages"]:
        speaker = "🧑" if isinstance(msg, HumanMessage) else "🤖"
        print(f"{speaker}: {msg.content}")
    print("-" * 40)


# -----------------------------
# 7. Interactive loop
# -----------------------------
if __name__ == "__main__":
    print("💬 Chatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        run_chatbot(user_input)
