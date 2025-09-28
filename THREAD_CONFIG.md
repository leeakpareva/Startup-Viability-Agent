# Thread Configuration for LangSmith

## Overview
NAVADA now supports conversation thread tracking through LangSmith, enabling persistent conversation history across multi-turn interactions. This allows the agent to maintain context throughout a conversation and provide more coherent, contextually-aware responses.

## Features

### 🧵 Thread Management
- **Automatic Session ID Generation**: Each new chat session gets a unique UUID
- **Thread Persistence**: Conversations are grouped by session ID in LangSmith
- **History Retrieval**: Previous messages are automatically retrieved for context

### 📊 LangSmith Tracing
- **Full Conversation Tracking**: All LLM interactions are traced and logged
- **Metadata Tagging**: Each trace includes session_id, project_name, and timestamps
- **Performance Monitoring**: Track latency, token usage, and success rates

### 🔄 Conversation Continuity
- **Context Preservation**: The agent remembers previous messages in the conversation
- **Persona Consistency**: Maintains investor/founder mode across the thread
- **Memory Integration**: Works with existing session memory features

## Setup

### 1. Get LangSmith API Key
1. Sign up for LangSmith at https://smith.langchain.com
2. Navigate to Settings → API Keys
3. Create a new API key

### 2. Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your keys:
OPENAI_API_KEY=your_openai_key_here
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=navada-startup-agent  # Optional: custom project name
```

### 3. Verify Installation
When you start the app, you should see:
```
✅ LangSmith tracing enabled
🧵 Thread initialized: abcd1234...
```

If LangSmith is not configured, you'll see:
```
ℹ️ LangSmith tracing disabled (no API key)
```

## Usage

### Automatic Thread Tracking
No code changes needed! Threads are automatically managed:

```python
# When a user starts a chat:
# 1. A unique session_id is generated (UUID4)
# 2. Thread metadata is initialized
# 3. All messages are tagged with the session_id
```

### Accessing Thread History
The agent automatically retrieves conversation history:

```python
# For each message:
# 1. Previous messages are fetched from LangSmith
# 2. Context is built from historical messages
# 3. Response considers full conversation context
```

### Example Conversation Flow
```
User: "Hi, my name is Bob"
Agent: "Hello Bob! I'm NAVADA, your startup viability agent."

User: "What is my name?"
Agent: "Your name is Bob, as you mentioned earlier."  # Remembers from thread history

User: "What was the first message I sent you?"
Agent: "The first message you sent was 'Hi, my name is Bob'."  # Full context awareness
```

## Implementation Details

### Key Functions

#### `get_thread_history(thread_id, project_name)`
Retrieves conversation history from LangSmith for a specific thread.

```python
def get_thread_history(thread_id: str, project_name: str) -> List[Dict[str, Any]]:
    # Filter runs by thread_id
    # Extract messages from LLM runs
    # Return chronological message history
```

#### `process_with_thread_context(question, session_id, get_chat_history, persona)`
Processes messages with full thread context and LangSmith tracing.

```python
@traceable(name="NAVADA Chat Pipeline")
def process_with_thread_context(...):
    # Retrieve historical messages
    # Build context with persona
    # Make API call with metadata
    # Return response
```

### Thread Metadata Structure
```python
{
    "session_id": "uuid-string",        # Unique conversation identifier
    "project_name": "navada-startup",   # LangSmith project
    "start_time": "2024-01-01T12:00:00" # ISO timestamp
}
```

### LangSmith Filters
The system uses specific metadata keys for filtering:
- `session_id`
- `conversation_id`
- `thread_id`

Any of these can be used to group traces into threads.

## Viewing Threads in LangSmith

### 1. Navigate to Threads Tab
In your LangSmith project, click on the "Threads" tab to see all conversations.

### 2. Thread List View
- Threads sorted by most recent activity
- Shows participant count and message count
- Click to view full conversation

### 3. Conversation View
- Chat-like UI showing input/output pairs
- Timestamps for each interaction
- Ability to annotate and provide feedback

### 4. Trace Details
Click "Open trace" to see:
- Token usage
- Latency metrics
- Full prompt/response
- Metadata and tags

## Best Practices

### 1. Session Management
- Let the system auto-generate session IDs
- Don't reuse session IDs across different users
- Store session IDs if you need to resume conversations

### 2. Error Handling
- The system gracefully degrades if LangSmith is unavailable
- Local memory still works without LangSmith
- Thread history retrieval has timeout protection

### 3. Performance
- Thread history is cached for the session
- Only LLM runs are retrieved (not all traces)
- History retrieval happens asynchronously

### 4. Privacy & Security
- Don't include sensitive data in conversations
- Use environment variables for API keys
- Consider data retention policies in LangSmith

## Troubleshooting

### Issue: "LangSmith tracing disabled"
**Solution**: Check that `LANGSMITH_API_KEY` is set in your `.env` file

### Issue: Thread history not working
**Solutions**:
1. Verify LangSmith API key is valid
2. Check project name matches in LangSmith dashboard
3. Ensure you have proper permissions in LangSmith

### Issue: Conversations not grouped
**Solution**: Verify metadata keys are correctly set (session_id, thread_id, or conversation_id)

### Issue: Slow response times
**Solutions**:
1. Check LangSmith API latency
2. Consider limiting history retrieval to recent messages
3. Use async operations where possible

## Advanced Configuration

### Custom Project Names
```python
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "navada-startup-agent")
```

### Filtering Thread History
```python
filter_string = f'and(in(metadata_key, ["session_id"]), eq(metadata_value, "{thread_id}"))'
```

### Extending Thread Metadata
```python
thread_metadata = {
    "session_id": session_id,
    "project_name": LANGSMITH_PROJECT,
    "start_time": timestamp,
    "user_id": user_id,        # Add custom fields
    "environment": "production"
}
```

## Integration with Existing Features

### Works With:
- ✅ Persona modes (Investor/Founder)
- ✅ Session memory
- ✅ Chart generation
- ✅ Web scraping
- ✅ Viability assessments
- ✅ All existing NAVADA features

### No Conflicts With:
- Local file operations
- CSV uploads
- Interactive dashboards
- PDF report generation

## Future Enhancements

Potential improvements to thread management:
1. **Thread Summarization**: Auto-summarize long conversations
2. **Thread Export**: Export conversation history to JSON/CSV
3. **Thread Analytics**: Analyze conversation patterns
4. **Multi-User Threads**: Support for collaborative sessions
5. **Thread Templates**: Pre-configured conversation flows

## Support

For issues or questions about thread configuration:
1. Check LangSmith documentation: https://docs.langsmith.com
2. Review the NAVADA logs for error messages
3. Ensure all dependencies are up-to-date