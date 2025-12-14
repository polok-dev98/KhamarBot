# chat_engine.py

import os
# REMOVE this line: from dotenv import load_dotenv
from modules.chat_history import get_chat_history, append_chat_history
from modules.agent import run_agent

# REMOVE this line: load_dotenv()

async def get_chat_response(user_query: str, session_id: str, user_id: str) -> str:
    """Get response using the agentic RAG system"""
    
    # Load chat history
    history_records = await get_chat_history(session_id)
    
    # Run the agent
    response = await run_agent(user_query, session_id, user_id, history_records)
    
    # Save to history
    await append_chat_history(session_id, user_query, response, user_id)
    
    return response