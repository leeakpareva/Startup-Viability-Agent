# 🚀 NAVADA - Startup Viability Agent

**AI-Powered Startup Analysis Platform with Voice Support & Real-Time Intelligence**

NAVADA is an advanced AI-powered conversational agent that helps investors and founders make data-driven decisions about startup viability using real-time market intelligence, voice interactions, and comprehensive analytics.

![Python](https://img.shields.io/badge/Python-3.8+-green) ![Chainlit](https://img.shields.io/badge/Framework-Chainlit-blue) ![OpenAI](https://img.shields.io/badge/AI-GPT--4-orange) ![LangSmith](https://img.shields.io/badge/Tracking-LangSmith-purple)

## 🎯 Key Features

### 🎭 **Dual AI Personas**
- **💼 Investor Mode**: VC perspective focused on ROI, market analysis, and investment opportunities
- **🚀 Founder Mode**: Entrepreneur perspective focused on execution, operations, and tactical recommendations
- **Smart Memory**: Conversation context preserved across sessions with LangSmith thread tracking

### 🔍 **Real-Time Market Intelligence**
- **Internet Search Integration**: Automatically searches for up-to-date market data and trends
- **Brave Search API**: Access to current startup news, funding rounds, and market intelligence
- **Context-Aware Analysis**: Search results tailored to your current persona and conversation

### 🎤 **Voice Interaction**
- **Text-to-Speech**: AI responses with natural voice output using OpenAI TTS
- **Voice Commands**: Simple "voice on/off" toggle for hands-free interaction
- **ElevenLabs Integration**: Advanced conversational AI widget support

### 🧮 **Mathematical Analysis Mode**
- **IRR/NPV Calculations**: Internal Rate of Return and Net Present Value for investment analysis
- **Revenue Projections**: Model growth scenarios with compound interest calculations
- **Monte Carlo Simulations**: Run 1,000+ scenarios for exit value predictions
- **Burn Rate Optimization**: Calculate optimal spending for desired runway
- **Financial Modeling**: Comprehensive startup metrics and break-even analysis

### 📊 **Advanced Analytics & Visualization**
- **24 Startup Dataset**: Comprehensive database with enhanced metrics (Business Model, Moat, MRR, Growth Rate)
- **6 Chart Types**: Growth trajectory, team performance, market opportunity, funding efficiency, stage progression, risk assessment
- **Interactive Dashboards**: Real-time data exploration with Plotly visualizations
- **PDF Reports**: Professional investment reports with executive summaries

### 🔧 **Enterprise Features**
- **LangSmith Observability**: Complete conversation tracking and analytics
- **Thread Management**: Persistent conversation history and context
- **LangChain Database**: Vector storage for enhanced RAG capabilities
- **Deployment Ready**: LangGraph compatibility for production scaling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (recommended 3.11+)
- OpenAI API key
- Optional: Brave Search API key, LangSmith API key

### Installation

```bash
# Clone repository
git clone <repository-url>
cd navada

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Setup

Create `.env` file with:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Enhanced Features
SEARCH_API_KEY=your_brave_search_api_key    # For real-time market intelligence
LANGSMITH_API_KEY=your_langsmith_api_key    # For conversation tracking
TTS_PROMPT_ID=your_tts_prompt_id           # For voice output
LANGCHAIN_DATABASE_ID=your_database_id     # For RAG capabilities
```

### Run Application

```bash
chainlit run app.py --host 0.0.0.0 --port 8000
```

Open browser to `http://localhost:8000`

## 🎮 How to Use

### 🎭 **Switch Personas**
```
investor mode    # Switch to VC perspective
founder mode     # Switch to entrepreneur perspective
```

### 🔍 **Real-Time Search**
```
search AI startups 2024
latest trends in fintech
current venture capital news
what's happening with [topic]
```

### 🎤 **Voice Control**
```
voice on         # Enable text-to-speech
voice off        # Disable text-to-speech
```

### 🧮 **Mathematical Analysis**
```
math mode        # Enter advanced calculation mode
calculate IRR for 5x return in 7 years
project revenue with 20% monthly growth
simulate 1000 scenarios for exit
optimize burn rate for 18 month runway
exit math mode   # Return to regular mode
```

### 📊 **Analytics & Charts**
```
show funding vs burn chart
display growth trajectory
generate portfolio analysis
create risk assessment
```

### 📄 **Reports**
```
generate report
portfolio analysis
```

## 🏗️ Tech Stack

### **Core Framework**
- **Chainlit 2.8.0**: Conversational AI interface
- **OpenAI GPT-4o-mini**: Natural language processing
- **Python 3.11+**: Backend runtime

### **AI & Analytics**
- **LangChain**: RAG and vector storage
- **LangSmith**: Conversation tracking and observability
- **LangGraph**: Production deployment support
- **Pandas/NumPy**: Data processing and analysis

### **Visualization**
- **Plotly**: Interactive charts and dashboards
- **Matplotlib/Seaborn**: Statistical visualizations
- **ReportLab**: PDF report generation

### **Real-Time Intelligence**
- **Brave Search API**: Live market data
- **BeautifulSoup**: Web scraping capabilities
- **OpenAI TTS**: Voice output
- **ElevenLabs**: Advanced voice AI

### **Data & Storage**
- **Chroma**: Vector database for RAG
- **Environment Variables**: Secure configuration
- **Session Management**: Persistent conversations

## 📊 Dataset

Enhanced startup dataset with 24 companies featuring:
- **Core Metrics**: Funding, burn rate, founder experience, market size
- **Business Intelligence**: Business model, competitive moat, MRR tracking
- **Growth Analytics**: Growth rate, team size, years since founding
- **Risk Assessment**: Competition analysis, stage progression

## 🔒 Security Features

- **API Key Protection**: Environment variable encryption
- **Secure Web Scraping**: URL validation and content filtering
- **Session Isolation**: User data separation
- **Rate Limiting**: API usage protection

## 🚀 Deployment

### LangGraph Deployment
```bash
# Deploy with LangGraph wrapper
langgraph deploy
```

### Docker Deployment
```bash
# Build container
docker build -t navada .

# Run container
docker run -p 8000:8000 navada
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: Built-in help system
- **Community**: GitHub Discussions

## 📄 License

MIT License - see LICENSE file for details.

---

**Built for the startup ecosystem with AI, voice, and real-time intelligence** 🌟

**Designed and Developed by Lee Akpareva MBA, MA**