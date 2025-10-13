# RAG Project - Intelligent Document Q&A System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines multiple AI agents to answer questions across different domains including mathematics, automotive data, and geographical information.

## 🚀 Features

- **Multi-Agent System**: Specialized agents for math, cars, countries, and general queries
- **LangGraph Workflow**: State-of-the-art orchestration framework for complex AI workflows
- **Vector Database**: ChromaDB for efficient document storage and semantic search
- **Interactive Chat Interface**: Chainlit-powered web UI for real-time interactions
- **Smart Caching**: Avoids re-ingestion of existing data for faster startup
- **Comprehensive Logging**: Detailed logging with Loguru for debugging and monitoring

## 🛠️ Technologies Used

- **LangChain**: Framework for building LLM-powered applications
- **LangGraph**: Advanced workflow orchestration for complex agent interactions
- **Chainlit**: Interactive chat interface for AI applications
- **ChromaDB**: High-performance vector database for document storage
- **OpenAI GPT-4**: Large language model for intelligent responses
- **Docling**: Document processing and chunking for RAG
- **uv**: Fast Python package manager and project management

## 📋 Prerequisites

- Python 3.11 or higher
- OpenAI API key (set in `.env` file)
- uv package manager

## ⚡ Quick Start

### Step 0: Environment Setup

1. **Clone and navigate to the project**:
   ```bash
   git clone <your-repo-url>
   cd RAG_Project
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Set up virtual environment and install dependencies**:
   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:
   ```bash
   uv shell
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key:
   # OPENAI_API_KEY=your_api_key_here
   ```

### Step 1: Data Ingestion

Run the main script to initialize the vector database:

```bash
python main.py
```

This will:
- Load and process car dataset from `data/cars_dataset.csv`
- Process country information from `data/country_data.md`
- Create vector embeddings and store in ChromaDB
- Skip ingestion if data already exists (smart caching)

### Step 2: Test the System

The script will automatically run evaluation queries to test functionality:

```bash
# Example queries that will be tested:
# - "What is the square root of 7?" (Math agent with LLMMathChain)
# - "What is capital of Oslo?" (Countries agent)
```

## 🎯 Usage Options

### Option 1: Command Line Interface

Run queries directly from Python:

```python
from src.workflow import process_query

# Process a query
result = process_query("What is the capital of Norway?")
print(result["response"])
```

### Option 2: Interactive Chat Interface (Recommended)

Start the Chainlit web interface:

```bash
chainlit run main.py -w
```

This opens a beautiful web interface at `http://localhost:8000` with:
- **Real-time chat** with your AI agents
- **Visual workflow tracing** showing agent routing
- **Interactive debugging** with LangChain callbacks
- **Persistent conversation** support

## 🏗️ Project Structure

```
RAG_Project/
├── main.py                 # Main entry point and Chainlit interface
├── src/
│   ├── agents.py           # Specialized AI agents (math, cars, countries)
│   ├── classifier.py       # Query classification logic
│   ├── ingestion.py        # Data loading and vectorization
│   ├── retrieval.py        # Vector search functionality
│   ├── workflow.py         # LangGraph workflow orchestration
│   ├── models.py           # LLM and embedding model configuration
│   └── logger.py           # Logging configuration
├── data/
│   ├── cars_dataset.csv    # Automotive dataset
│   └── country_data.md     # Geographical information
├── chroma_db/              # Vector database (auto-generated)
├── outputs/                # Logs and workflow diagrams
└── pyproject.toml          # Project dependencies and configuration
```

## 🤖 Agent System

### Available Agents:

1. **Math Agent**: Uses `LLMMathChain` for complex mathematical calculations
2. **Cars Agent**: Provides detailed information about vehicles from the dataset
3. **Countries Agent**: Answers geographical and country-specific questions
4. **General Agent**: Handles queries that don't fit other categories

### Query Routing:
- **Classifier**: Analyzes queries and routes to appropriate agent
- **Workflow**: Orchestrates the complete request-response cycle
- **Format Agent**: Ensures consistent response formatting

## 🔧 Configuration

### Environment Variables (`.env`):
```env
OPENAI_API_KEY=your_openai_api_key
```

### Model Settings (`src/models.py`):
- **LLM**: GPT-4o-mini for fast, cost-effective responses
- **Embeddings**: text-embedding-3-small for efficient vectorization

## 📊 Performance Features

- **Smart Ingestion Check**: Avoids re-processing existing data
- **Efficient Vector Search**: Optimized retrieval with metadata filtering
- **Streaming Responses**: Real-time response generation in Chainlit UI
- **Comprehensive Logging**: Detailed logs in `./outputs/loguru.log`

## 🚨 Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure virtual environment is activated (`uv shell`)
2. **OpenAI API Issues**: Check API key in `.env` file
3. **Missing Data**: Run `python main.py` to initialize database
4. **Chainlit Not Starting**: Install with `uv add chainlit` if needed

### Logs Location:
- Main logs: `./outputs/loguru.log`
- ChromaDB logs: `./chroma_db/` directory

## 🎯 Example Queries

Try these queries in the Chainlit interface:

**Math Queries:**
- "What is the square root of 144?"
- "Calculate 15% of 350"
- "What is (25 × 4) + 18?"

**Car Queries:**
- "Show me cars launched in 2020"
- "What are the safest cars available?"
- "Find me electric vehicles with high user ratings"

**Country Queries:**
- "What is the capital of Norway?"
- "Tell me about the geography of Japan"
- "Which countries border France?"

## 🔄 Development

### Adding New Agents:
1. Create agent function in `src/agents.py`
2. Update classifier in `src/classifier.py`
3. Add routing in `src/workflow.py`

### Custom Data Sources:
1. Add data files to `data/` directory
2. Create ingestion function in `src/ingestion.py`
3. Update `vector_store_setup()` function

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced query preprocessing
- [ ] Custom embedding models
- [ ] API endpoints for external integrations
- [ ] Advanced analytics and metrics

---

**Happy querying! 🚀**