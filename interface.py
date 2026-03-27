import sys
import signal
from typing import Optional
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama for cross-platform colored output
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

from model_loader import ModelLoader
from chat_memory import ChatMemory

class ChatInterface:
    
    def __init__(self, model_name="gpt2", memory_window=4):
        self.model_loader = ModelLoader(model_name)
        self.memory = ChatMemory(memory_window)
        self.running = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self._print_colored("\n\nInterrupted by user. Exiting gracefully...", "yellow")
        self._exit_chatbot()
    
    def _print_colored(self, text: str, color: str = "white", end: str = "\n"):
        """Print colored text if colorama is available."""
        if not COLORS_AVAILABLE:
            print(text, end=end)
            return
            
        color_map = {
            "red": Fore.RED,
            "green": Fore.GREEN,
            "yellow": Fore.YELLOW,
            "blue": Fore.BLUE,
            "magenta": Fore.MAGENTA,
            "cyan": Fore.CYAN,
            "white": Fore.WHITE,
            "bright_green": Fore.LIGHTGREEN_EX,
            "bright_blue": Fore.LIGHTBLUE_EX,
        }
        
        color_code = color_map.get(color, Fore.WHITE)
        print(f"{color_code}{text}{Style.RESET_ALL}", end=end)
    
    def _print_banner(self):
        """Print welcome banner."""
        banner = """
***************************************************************
*                   Local Chatbot Assistant                   *
*                   Powered by Hugging Face                   *
***************************************************************
        """
        self._print_colored(banner, "cyan")
        self._print_colored("Type '/exit' to quit, '/clear' to clear memory, '/help' for commands\n", "yellow")
    
    def _print_help(self):
        """Print available commands."""
        help_text = """
Available commands:
  /exit    - Exit the chatbot
  /clear   - Clear conversation memory
  /help    - Show this help message
  /status  - Show memory and model status
  /history - Show recent conversation history
        """
        self._print_colored(help_text, "blue")
    
    def _show_status(self):
        """Show current status of memory and model."""
        memory_info = self.memory.get_memory_summary()
        model_status = "Loaded" if self.model_loader.is_loaded() else "Not Loaded"
        
        status_text = f"""
Status Information:
  Model: {self.model_loader.model_name} ({model_status})
  Device: {self.model_loader.device}
  Memory Window Size: {memory_info['window_size']}
  Current Exchanges: {memory_info['current_exchanges']}
  Total Exchanges: {memory_info['total_exchanges']}
  Memory Full: {memory_info['memory_full']}
        """
        self._print_colored(status_text, "blue")
    
    def _show_history(self):
        """Show recent conversation history."""
        if not self.memory.has_context():
            self._print_colored("No conversation history available.", "yellow")
            return
        
        history = self.memory.get_recent_context()
        self._print_colored("\nRecent Conversation History:", "blue")
        self._print_colored("-" * 50, "blue")
        
        for i, (user_msg, bot_msg) in enumerate(history, 1):
            self._print_colored(f"[{i}] User: {user_msg}", "white")
            self._print_colored(f"[{i}] Bot:  {bot_msg}", "bright_green")
            print()
    
    def _handle_command(self, user_input: str) -> bool:
        command = user_input.strip().lower()
        
        if command == "/exit":
            self._exit_chatbot()
            return True
        elif command == "/clear":
            self.memory.clear_memory()
            self._print_colored("Conversation memory cleared!", "green")
            return True
        elif command == "/help":
            self._print_help()
            return True
        elif command == "/status":
            self._show_status()
            return True
        elif command == "/history":
            self._show_history()
            return True
        
        return False
    
    def _exit_chatbot(self):
        """Exit the chatbot"""
        self.running = False
        self._print_colored("Exiting chatbot. Goodbye! 👋", "green")
        sys.exit(0)
    
    def _get_user_input(self) -> Optional[str]:
        """Get user input with error handling."""
        try:
            if COLORS_AVAILABLE:
                return input(f"{Fore.BLUE}User: {Style.RESET_ALL}")
            else:
                return input("User: ")
        except (EOFError, KeyboardInterrupt):
            return None
    
    def initialize(self) -> bool:
        """Initialize the chatbot by loading the model."""
        self._print_colored("Initializing chatbot...", "yellow")
        
        if not self.model_loader.load_model():
            self._print_colored("Failed to load model. Please check your internet connection and try again.", "red")
            return False
        
        self._print_colored("Chatbot initialized successfully! Ready to assist you 🚀", "green")
        return True
    
    def run(self):
        if not self.initialize():
            return
        
        self._print_banner()
        
        self.running = True
        
        while self.running:
            try:
                user_input = self._get_user_input()
                
                if user_input is None:
                    self._exit_chatbot()
                    break
                
                user_input = user_input.strip()
                
                if not user_input:
                    continue
                if self._handle_command(user_input):
                    continue
                
                self._print_colored("Bot: ", "bright_green", end="")
                print("Thinking...", end="", flush=True)
                
                try:
                    context_prompt = self.memory.get_context_prompt(user_input)                   
                    response = self.model_loader.generate_response(context_prompt)
                    print("\r" + " " * 12 + "\r", end="")  
                    self._print_colored(f"Bot:  {response}", "bright_green")
                    
                    # Add to memory
                    self.memory.add_exchange(user_input, response)
                    
                except Exception as e:
                    print("\r" + " " * 12 + "\r", end="")  
                    self._print_colored(f"Bot:  Sorry, I encountered an error: {str(e)}", "red")
                
                print() 
                
            except Exception as e:
                self._print_colored(f"Unexpected error: {str(e)}", "red")
                continue

def main():
    chatbot = ChatInterface(
        model_name="gpt2",  
        memory_window=4  
    )
    
    try:
        chatbot.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
