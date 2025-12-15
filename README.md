# 🐄 KhamarBot - Domestic Animals Help Provider Bot

KhamarBot is an **Agentic RAG (Retrieval-Augmented Generation)** system designed to provide intelligent assistance for domestic animal and livestock care. The bot uses advanced AI with tool calling, self-reflection, and knowledge retrieval to deliver accurate, practical advice for farmers and livestock owners.

## 🚀 Features

- **🤖 Intelligent Agentic Architecture**: Uses LangGraph for state management and decision routing
- **🔍 Semantic Search**: Cross-lingual retrieval (Bengali ↔ English) using Azure OpenAI embeddings
- **🤔 Self-Reflection**: Built-in critic agent evaluates retrieved information quality
- **📚 Knowledge Base**: Vector database from PDF-extracted livestock documents in Bengali
- **🌐 Web Interface**: Flask-based web application with session management
- **💬 Conversation History**: Persistent chat history with timestamps

## 🏗️ Architecture

```
┌─────────────────┐    HTTP    ┌─────────────┐    Async    ┌─────────────┐
│   Web Browser   │ ◀────────▶ │   Flask     │ ◀────────▶ │   Agent     │
│   (User)        │   JSON     │   (app.py)  │   Calls    │   System    │
└─────────────────┘            └─────────────┘            └──────┬──────┘
                                                                  │
                                                     ┌────────────┼────────────┐
                                                     │           │            │
                                               ┌─────▼─────┐ ┌──▼──┐ ┌───────▼──────┐
                                               │  Retriever │ │Critic│ │   Answer    │
                                               │   Tool     │ │Agent │ │  Generator  │
                                               └────────────┘ └──────┘ └─────────────┘
                                                     │
                                              ┌──────▼──────┐
                                              │  Vector DB  │
                                              │ (FAISS)     │
                                              └─────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Azure OpenAI account with API keys
- Git

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/polok-dev98/KhamarBot.git
cd KhamarBot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_KEY=your_azure_openai_key_here
AZURE_OPENAI_URL=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini

# Azure Embeddings Configuration
AZURE_EMBEDDING_KEY=your_azure_embedding_key_here
AZURE_EMBEDDING_URL=https://your-resource.cognitiveservices.azure.com/
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

```


### 5.Run the App
```bash
python app.py
```

## 📁 Project Structure

```
KhamarBot/
├── app.py                      # Flask web server
├── modules/
│   ├── chat_engine.py          # Chat orchestration
│   ├── agent.py               # LangGraph agent system
│   ├── retriever.py           # Vector search tool
│   ├── chat_history.py        # Conversation storage
│   └── index_builder.py       # PDF-to-vector processing
├── templates/
│   └── index.html             # Web interface
├── static/                    # CSS/JS files
├── pdf_file/                  # PDF-extracted JSON files
├── vector_store/              # FAISS index & metadata
├── data/                      # Additional data files
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Setup Instructions

### Step 1: Prepare Knowledge Base
Place your PDF-extracted JSON files in the `pdf_file/` directory. JSON format should be:
```json
[
  {
    "content": "বাংলা পাঠ্য বিষয়বস্তু...",
    "header": "শিরোনাম",
    "page": "পৃষ্ঠা নম্বর"
  }
]
```

### Step 2: Build Vector Database
```bash
python index_builder.py
```
This will:
- Process all JSON files in `pdf_file/`
- Split text into chunks (800 chars with 100 overlap)
- Generate embeddings using Azure OpenAI
- Create FAISS index and save to `vector_store/`

### Step 3: Run the Application
```bash
python app.py
```
The server will start at `http://localhost:5000`

### Step 4: Access the Web Interface
1. Open your browser to `http://localhost:5000`
2. Start a new session
3. Ask questions about livestock care in English or Bengali

## 🔄 Update Knowledge Base

To add new documents:
1. Place new JSON files in `pdf_file/`
2. Rebuild the index:
   ```bash
   python index_builder.py
   ```
3. Restart the application

## 🎯 Usage Examples

### English Queries:
- "How to treat cattle fever?"
- "Best food for dairy cows"
- "Chicken vaccination schedule"
- "Prevent foot and mouth disease"

### Bengali Queries:
- "গরুর জ্বর হলে কি করব?"
- "ডেইরি গরুর জন্য সেরা খাদ্য"
- "মুরগির টিকার সময়সূচী"
- "খুরা রোগ প্রতিরোধ"

## ⚡ API Endpoints

### Start a Session
```bash
POST /start
{
  "user_id": "optional_user_id"
}
```
Response:
```json
{
  "message": "Session started.",
  "user_id": "user123",
  "session_id": "user123_abc456"
}
```

### Chat
```bash
POST /chat
{
  "message": "How to treat cattle fever?",
  "session_id": "user123_abc456",
  "user_id": "user123"
}
```
Response:
```json
{
  "response": "For cattle fever, first isolate the infected animal...",
  "session_id": "user123_abc456"
}
```

## 🧠 How It Works

### 1. Query Processing
- User submits query (English or Bengali)
- Agent router determines processing path
- Greetings/simple queries get direct responses
- Complex queries trigger full RAG pipeline

### 2. Knowledge Retrieval
- Query converted to semantic embedding
- FAISS vector search finds relevant Bengali documents
- Results shown in terminal with source metadata

### 3. Self-Reflection
- Critic agent evaluates retrieved information
- Checks relevance, completeness, safety
- Identifies gaps in knowledge

### 4. Answer Generation
- Combines query, retrieved context, and critic evaluation
- Generates practical, safety-conscious advice
- References source materials when possible

## 📊 Terminal Debug Output

When running, you'll see real-time debugging:
```
[Agent] Processing query: 'How to treat cattle fever?'
🔍 [RETRIEVER DEBUG] Query: 'How to treat cattle fever?'
✅ [RETRIEVER DEBUG] Found 3 relevant documents:
   📄 Document 1 (Similarity: 0.85)
      Topic: Cattle Fever Treatment
      Book: livestock_health_guide.pdf.json
      Page: 34
      Content: Treatment involves isolating the infected animal...
[Agent] Critic evaluating livestock information quality...
[Agent] Generating livestock care answer...
```

## 🔧 Configuration Options

### Retriever Settings (`retriever.py`):
```python
# Adjust these parameters:
top_k = 5                    # Number of documents to retrieve
similarity_threshold = 0.4   # Minimum similarity score (0-1)
```

### Text Chunking (`index_builder.py`):
```python
CHUNK_SIZE = 800      # Characters per chunk
CHUNK_OVERLAP = 100   # Overlap between chunks
```

### Agent Settings (`agent.py`):
```python
temperature = 0.1     # LLM creativity (0=deterministic, 1=creative)
history_length = 6    # Conversation history to include
```

