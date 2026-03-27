# Local Command-Line Chatbot with Hugging Face

A fully functional local chatbot interface using Hugging Face text generation models with conversation memory and a robust CLI experience.

## Features

- **Local Model Execution**: Runs entirely on your machine using Hugging Face transformers
- **Conversation Memory**: Maintains context using a sliding window buffer (configurable size)
- **Modular Architecture**: Clean, maintainable code structure with separate concerns
- **Robust CLI Interface**: User-friendly command-line interface with colored output
- **Graceful Error Handling**: Comprehensive error handling and recovery
- **Multi-Platform Support**: Works on Windows, macOS, and Linux
- **GPU Acceleration**: Automatically detects and uses available GPU resources

## Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for better performance)
- Internet connection for initial model download
- Optional: CUDA-compatible GPU for faster inference

## 🛠️ Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd huggingface-chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Quick Start

```bash
python main.py
```

### Alternative Entry Point

```bash
python interface.py
```

## Usage Examples

### Basic Conversation

```
User: What is the capital of France?
Bot:  The capital of France is Paris.

User: And what about Italy?
Bot:  The capital of Italy is Rome.

User: Tell me something interesting about Rome.
Bot:  Rome is known as the "Eternal City" and has over 2,500 years of history.
```

### Available Commands

- `/exit` - Exit the chatbot
- `/clear` - Clear conversation memory
- `/help` - Show available commands
- `/status` - Show memory and model status
- `/history` - Show recent conversation history

### Command Examples

```
User: /status
Status Information:
  Model: microsoft/DialoGPT-small (Loaded)
  Device: cuda
  Memory Window Size: 4
  Current Exchanges: 2
  Total Exchanges: 2
  Memory Full: False

User: /history
Recent Conversation History:
--------------------------------------------------
[1] User: What is the capital of France?
[1] Bot:  The capital of France is Paris.

[2] User: And what about Italy?
[2] Bot:  The capital of Italy is Rome.

User: /exit
Exiting chatbot. Goodbye! 👋
```

##  Architecture

The project follows a modular architecture with clear separation of concerns:

### Core Modules

1. **`model_loader.py`** - Model and tokenizer management
   - Handles Hugging Face model loading
   - Manages GPU/CPU device selection
   - Provides text generation interface

2. **`chat_memory.py`** - Conversation memory buffer
   - Implements sliding window mechanism
   - Manages conversation context
   - Provides history export functionality

3. **`interface.py`** - CLI interface and integration
   - User interaction handling
   - Command processing
   - Colored output and formatting

4. **`main.py`** - Application entry point

### Key Design Decisions

- **Sliding Window Memory**: Keeps last N conversation turns to maintain context while preventing memory bloat
- **Modular Design**: Each component has a single responsibility and can be tested independently
- **Error Resilience**: Comprehensive error handling ensures the chatbot continues running despite individual failures
- **Device Agnostic**: Automatically detects and uses the best available compute device (GPU/CPU)

##  Configuration

### Model Selection

You can change the model by modifying the `model_name` parameter in `interface.py`:

```python
chatbot = ChatInterface(
    model_name="GPT-2",  
    memory_window=4 
)
```

### Recommended Models

- **microsoft/DialoGPT-small** (Default) - Fast, lightweight
- **microsoft/DialoGPT-medium** - Better quality, slower
- **gpt2** - General-purpose text generation
- **distilgpt2** - Faster, smaller version of GPT-2

### Memory Window Size

Adjust the conversation memory window:

```python
memory_window=4  
memory_window=6 
```

##  Testing

### Manual Testing

1. Start the chatbot: `python main.py`
2. Test basic conversation flow
3. Verify memory persistence across multiple exchanges
4. Test all available commands
5. Test graceful exit with `/exit` or Ctrl+C

### Memory Window Testing

```
# Test conversation continuity
User: My name is Alice
Bot: Nice to meet you, Alice!

User: What's my name?
Bot: Your name is Alice.

# Test memory window limit (after 4+ exchanges)
User: Do you remember my name?
Bot: [Should remember if within window, forget if beyond]
```

## 📜 License

This project is created for educational purposes as part of the ATG Technical Assignment.
