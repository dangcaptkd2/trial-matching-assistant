# Clinical Trial Assistant

AI-powered assistant for clinical trial matching using LangGraph, Elasticsearch, and OpenAI.

## Features

### Core Capabilities

- [x] **Clinical Trial Matching**: Find suitable clinical trials based on patient profile (age, gender, medical condition, stage, location, etc.)
- [ ] **Trial Eligibility Check**: Check profile compatibility with a specific clinical trial by verifying inclusion and exclusion criteria
- [ ] **Inclusion/Exclusion Criteria Explanation**: Explain trial requirements in clear, understandable language
- [ ] **Clinical Trial Summarization**: Summarize clinical trials in a concise, easy-to-understand format with key information
- [ ] **Medical Terminology Explanation**: Explain medical terminology found in clinical trial documents in plain language
- [ ] **Trial Comparison**: Compare related clinical trials for a specific disease, highlighting differences in eligibility, treatment, and locations

### Technical Features

- **Intelligent Routing**: Automatically routes queries to appropriate handlers
- **LLM Reranking**: Uses GPT models to rerank search results
- **Conversation Memory**: Maintains context across multiple messages
- **OpenAI-Compatible API**: Works with Open WebUI and other OpenAI-compatible clients

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200
ES_INDEX_NAME=trec2023_ctnlp

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
STREAMING_ENABLED=true  # Enable/disable streaming responses globally

# LLM Settings
LLM_MODEL=gpt-4.1-nano
TEMPERATURE=0.0

# LangSmith (optional, for monitoring)
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=clinical-trial-matching
LANGCHAIN_TRACING_V2=true
```

### 3. Start the API

After running `uv sync`, install the package in editable mode to enable the `api` command:

```bash
uv pip install -e .
uv run api
```

Or run directly without installing:

```bash
uv run python -m src.api
```

The API will start on `http://localhost:8000`

### 4. Use Open WebUI (Recommended)

```bash
./start_open_webui.sh
```

Or manually:

```bash
docker run -d \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Then open `http://localhost:3000`, create an account, and select model: `clinical-trial-assistant`

## Testing

### Test API

```bash
python test_openai_api.py
```

### Test with curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clinical-trial-assistant",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### Test Workflow

```bash
python scripts/test_workflow.py --input "Find trials for breast cancer"
```

## API Endpoints

### OpenAI-Compatible
- `POST /v1/chat/completions` - Chat completions (streaming & non-streaming)
- `GET /v1/models` - List available models

### Management
- `GET /api/health` - Health check
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation details
- `DELETE /api/conversations/{id}` - Delete conversation

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## Workflow

The assistant uses a LangGraph workflow:

1. **Reception**: Classifies user input (chitchat, patient matching, or trial lookup)
2. **Search**: Searches Elasticsearch for relevant trials
3. **Rerank**: Uses LLM to rerank results based on patient profile
4. **Synthesize**: Creates natural language response

For trial lookup:
1. **Reception**: Extracts trial IDs
2. **Lookup**: Fetches trial details from Elasticsearch
3. **Synthesize**: Formats trial information

## Configuration

All configuration is in `src/config/settings.py` and can be overridden via environment variables in `.env`.

Key settings:
- `llm_model`: LLM model to use (default: `gpt-4.1-nano`)
- `es_index_name`: Elasticsearch index name
- `streaming_enabled`: Enable/disable streaming responses globally (default: `true`)

## Development

### Visualize Workflow

```bash
python scripts/visualize_graph.py
```

Output: `drafts/workflow_graph.png`

### Monitor with LangSmith

View traces at: https://smith.langchain.com

## Troubleshooting

**API won't start**
- Check Elasticsearch is running and accessible
- Verify OpenAI API key is set
- Check port 8000 is available

**Open WebUI can't connect**
- Make sure API is running: `curl http://localhost:8000/health`
- Use `host.docker.internal` instead of `localhost` in Docker
- On Linux, add `--add-host=host.docker.internal:host-gateway` to docker run

**No search results**
- Verify Elasticsearch index name matches
- Check index contains documents
- Review LangSmith traces for errors

## License

MIT
