# agent.py - Domestic Animals Help Provider Bot

import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from modules.retriever import VectorRetriever

# Define the state
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    user_query: str
    retrieved_context: str
    reflection: str
    final_answer: str
    session_id: str
    user_id: str

# Initialize retriever
retriever = VectorRetriever(
    index_path="vector_store/kb_index.faiss",
    metadata_path="vector_store/kb_metadata.pkl"
)

# Create a function to get LLM instance
def get_llm():
    """Lazy initialization of LLM"""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        azure_endpoint=os.getenv("AZURE_OPENAI_URL"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-05-15",
        temperature=0.1
    )

def retrieve_tool(state: AgentState) -> AgentState:
    """Tool: Retrieve relevant documents from knowledge base"""
    print(f"[Agent] Retrieving livestock information for: {state['user_query']}")
    context = retriever.as_tool(state['user_query'])
    return {**state, "retrieved_context": context}

def reflect_critic(state: AgentState) -> AgentState:
    """Critic: Reflect on the retrieved information and query"""
    print("[Agent] Critic evaluating livestock information quality...")
    
    reflection_prompt = f"""
    You are a veterinary expert evaluating information about domestic animals and livestock.
    
    User's Question: {state['user_query']}
    
    Retrieved Livestock Knowledge Base Information:
    {state['retrieved_context']}
    
    Analyze the following as a veterinary expert:
    1. How relevant and accurate is the retrieved information for answering the user's question?
    2. Are there any gaps in the information about animal care, treatment, or management?
    3. Is the information sufficient to provide a safe and helpful answer?
    4. What additional veterinary advice or warnings should be considered?
    5. Is the information specific enough to the animal type mentioned?
    
    Provide a concise veterinary evaluation that will help generate a responsible answer.
    """
    
    llm = get_llm()  # Get LLM instance here
    messages = [
        SystemMessage(content="You are a veterinary expert evaluating livestock and domestic animal care information."),
        HumanMessage(content=reflection_prompt)
    ]
    
    response = llm.invoke(messages)
    reflection = response.content
    
    # Add reflection to messages
    state['messages'].append({
        "role": "veterinary_expert",
        "content": f"[VETERINARY EVALUATION]: {reflection}"
    })
    
    return {**state, "reflection": reflection}

def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer based on query, context, and reflection"""
    print("[Agent] Generating livestock care answer...")
    
    # Prepare conversation history
    history_messages = [
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in state['messages'] 
        if not msg['content'].startswith('[VETERINARY EVALUATION]')
    ]
    
    history_text = "\n".join(history_messages[-6:])  # Last 6 messages
    
    answer_prompt = f"""
    You are a Domestic Animals Help Provider Bot specializing in livestock and farm animals.
    
    IMPORTANT INSTRUCTIONS:
    1. If the user greets you (e.g., 'hi', 'hello'), respond with the standard greeting
    2. Use the retrieved livestock knowledge base as your PRIMARY source
    3. Consider the veterinary expert's evaluation for safety and completeness
    4. If information is insufficient, recommend consulting a local veterinarian
    5. Format answers with clear, practical steps for farmers/livestock owners
    6. Include important safety warnings when discussing treatments or medications
    7. Reference source information (book name, page) when possible
    8. Be helpful, practical, and focused on Bangladeshi/Bengali livestock context
    
    STANDARD GREETING:
    "Hello! I'm your Domestic Animals Help Provider. I can assist with livestock care, animal health, farming practices, and veterinary advice for cattle, goats, chickens, and other farm animals. How can I help you today?"
    
    VETERINARY EXPERT'S EVALUATION:
    {state['reflection']}
    
    RETRIEVED LIVESTOCK KNOWLEDGE BASE:
    {state['retrieved_context']}
    
    CONVERSATION HISTORY:
    {history_text}
    
    USER'S QUESTION ABOUT ANIMALS:
    {state['user_query']}
    
    Provide your helpful, practical answer for livestock care:
    """
    
    llm = get_llm()  # Get LLM instance here
    messages = [
        SystemMessage(content="""You are a helpful Domestic Animals Help Provider specializing in livestock care. 
        You provide practical, safe advice for farmers and livestock owners based on veterinary knowledge.
        Always prioritize animal welfare and recommend professional veterinary care when needed."""),
        HumanMessage(content=answer_prompt)
    ]
    
    response = llm.invoke(messages)
    final_answer = response.content
    
    # Add final answer to messages
    state['messages'].append({
        "role": "animal_care_assistant",
        "content": final_answer
    })
    
    return {**state, "final_answer": final_answer}

def generate_greeting(state: AgentState) -> AgentState:
    """Generate greeting response without retrieval"""
    greeting = "Hello! I'm your Domestic Animals Help Provider. I can assist with livestock care, animal health, farming practices, and veterinary advice for cattle, goats, chickens, and other farm animals. How can I help you today?"
    
    state['messages'].append({
        "role": "animal_care_assistant",
        "content": greeting
    })
    
    return {**state, "final_answer": greeting}

def generate_simple_response(state: AgentState) -> AgentState:
    """Generate response for simple queries without retrieval"""
    user_query = state['user_query'].lower().strip()
    
    if user_query.startswith(('thanks', 'thank you')):
        response = "You're welcome! I'm happy to help with your livestock questions. Is there anything else about animal care I can assist you with?"
    elif user_query.startswith(('bye', 'goodbye')):
        response = "Goodbye! Wishing you and your animals good health. Feel free to reach out if you have more livestock questions."
    else:
        response = "Got it. Please let me know how I can help with your livestock or animal care questions."
    
    state['messages'].append({
        "role": "animal_care_assistant",
        "content": response
    })
    
    return {**state, "final_answer": response}

def should_use_tools(state: AgentState) -> str:
    """Router: Decide whether to use retrieval tools or not"""
    user_query = state['user_query'].lower().strip()
    
    # Greetings don't need retrieval
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'assalamualaikum']
    if any(user_query.startswith(greet) for greet in greetings):
        return "generate_greeting"
    
    # Simple queries don't need retrieval
    simple_queries = ['thanks', 'thank you', 'bye', 'goodbye', 'ok', 'okay', 'alright']
    if any(user_query.startswith(sq) for sq in simple_queries):
        return "generate_simple_response"
    
    # For animal/livestock related queries, use the full RAG pipeline
    return "retrieve"

# Build the agent graph
def create_agent_workflow():
    """Create and compile the agent workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", lambda state: state)  # Router node
    workflow.add_node("retrieve", retrieve_tool)
    workflow.add_node("reflect", reflect_critic)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("generate_greeting", generate_greeting)
    workflow.add_node("generate_simple_response", generate_simple_response)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        should_use_tools,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
            "generate_greeting": "generate_greeting",
            "generate_simple_response": "generate_simple_response"
        }
    )
    
    # Add edges for RAG pipeline
    workflow.add_edge("retrieve", "reflect")
    workflow.add_edge("reflect", "generate_answer")
    
    # Add edges to end
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("generate_greeting", END)
    workflow.add_edge("generate_simple_response", END)
    
    return workflow.compile()

# Create the compiled agent
agent = create_agent_workflow()

async def run_agent(user_query: str, session_id: str, user_id: str, history: List[Dict] = None) -> str:
    """Run the agentic RAG pipeline for livestock assistance"""
    
    # Prepare initial messages from history
    messages = []
    if history:
        for msg in history[-4:]:  # Last 4 messages for context
            messages.append({
                "role": "user" if "user" in msg else "assistant",
                "content": msg.get("user") or msg.get("bot", "")
            })
    
    # Add current query
    messages.append({
        "role": "user",
        "content": user_query
    })
    
    # Prepare initial state
    initial_state = AgentState(
        messages=messages,
        user_query=user_query,
        retrieved_context="",
        reflection="",
        final_answer="",
        session_id=session_id,
        user_id=user_id
    )
    
    # Run the agent
    print(f"[Animal Care Agent] Processing query: {user_query}")
    result = agent.invoke(initial_state)
    
    return result["final_answer"]