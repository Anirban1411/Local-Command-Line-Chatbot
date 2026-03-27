"""
chat_memory.py
Manages conversation history using a sliding window buffer mechanism.
"""

from collections import deque
from typing import List, Tuple, Optional

class ChatMemory:
    """Manages conversation history with a sliding window buffer."""
    
    def __init__(self, window_size=4):
        """
        Initialize the chat memory buffer.
        
        Args:
            window_size (int): Number of conversation turns to remember
        """
        self.window_size = window_size
        self.conversation_history = deque(maxlen=window_size)
        self.full_history = []  # Keep full history for potential future use
        
    def add_exchange(self, user_input: str, bot_response: str):
        """
        Add a user-bot exchange to the memory buffer.
        
        Args:
            user_input (str): User's message
            bot_response (str): Bot's response
        """
        exchange = (user_input, bot_response)
        self.conversation_history.append(exchange)
        self.full_history.append(exchange)
    
    def get_context_prompt(self, current_input: str) -> str:
        """
        Build a context-aware prompt using conversation history.
        
        Args:
            current_input (str): Current user input
            
        Returns:
            str: Formatted prompt with conversation context
        """
        if not self.conversation_history:
            return f"Human: {current_input}\nAssistant:"
        
        # Build conversation context
        context_parts = []
        
        for user_msg, bot_msg in self.conversation_history:
            context_parts.append(f"Human: {user_msg}")
            context_parts.append(f"Assistant: {bot_msg}")
        
        # Add current input
        context_parts.append(f"Human: {current_input}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def get_recent_context(self, num_turns: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Get recent conversation turns.
        
        Args:
            num_turns (int, optional): Number of recent turns to retrieve
            
        Returns:
            List[Tuple[str, str]]: List of (user_input, bot_response) tuples
        """
        if num_turns is None:
            return list(self.conversation_history)
        
        return list(self.conversation_history)[-num_turns:]
    
    def clear_memory(self):
        """Clear the conversation memory buffer."""
        self.conversation_history.clear()
        self.full_history.clear()
    
    def get_memory_summary(self) -> dict:
        """
        Get a summary of the current memory state.
        
        Returns:
            dict: Memory statistics and information
        """
        return {
            "window_size": self.window_size,
            "current_exchanges": len(self.conversation_history),
            "total_exchanges": len(self.full_history),
            "memory_full": len(self.conversation_history) == self.window_size
        }
    
    def export_conversation(self) -> List[Tuple[str, str]]:
        """
        Export the full conversation history.
        
        Returns:
            List[Tuple[str, str]]: Complete conversation history
        """
        return self.full_history.copy()
    
    def has_context(self) -> bool:
        """Check if there's any conversation context available."""
        return len(self.conversation_history) > 0