# 🚀 NAVADA - Startup Viability Agent

**AI-Powered Startup Analysis Platform with Interactive Dashboards & Web Scraping**

NAVADA (New Analysis & Viability Assessment for Data-driven Analysis) is a comprehensive AI-powered chatbot that helps investors, founders, and analysts make data-driven decisions about startup viability and risk assessment.

![NAVADA Interface](https://img.shields.io/badge/Framework-Chainlit-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![GPT-4](https://img.shields.io/badge/AI-GPT--4-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 **Key Features**

### 📊 **Interactive Dashboards**
- **Plotly-powered charts** with zoom, hover, and click functionality
- **Multi-panel sector dashboards** for comprehensive analysis
- **Interactive timelines** showing failure progression
- **Real-time data exploration** with export capabilities

### 🎭 **AI Personas & Memory**
- **💼 Investor Mode**: VC perspective focused on ROI and exit strategies
- **🚀 Founder Mode**: Entrepreneur perspective focused on execution
- **Session memory** remembers conversation history for context-aware responses
- **Persona-specific guidance** and recommendations

### 🔍 **Web Scraping & Analysis**
- **Smart web scraping** with CSS selector support
- **AI-powered website analysis** tailored to your persona
- **Competitor research** and market intelligence
- **Safety measures**: URL validation, content limits, timeout protection

### 📈 **Advanced Analytics**
- **8-factor viability scoring** with comprehensive metrics
- **Founder idea benchmarking** against dataset percentiles
- **Portfolio heatmap analysis** for multiple startups
- **Auto-generated insights** with risk detection and recommendations

### 📄 **Professional Reports**
- **PDF investment reports** with executive summaries
- **Comprehensive analysis** including charts and recommendations
- **Single startup** or **full portfolio** report generation
- **Professional formatting** ready for investor presentations

## 🎮 **Quick Start**

### Prerequisites
- Python 3.8+ (recommended 3.11+)
- OpenAI API key
- 2GB+ RAM for optimal performance

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/leeakpareva/Startup-Viability-Agent.git
cd Startup-Viability-Agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run NAVADA**
```bash
chainlit run app.py
```

5. **Open in browser**
- Navigate to `http://localhost:8000`
- Start analyzing startups! 🚀

## 📋 **Command Reference**

### 🎭 **Personas & Modes**
```bash
investor mode          # Switch to VC perspective
founder mode          # Switch to entrepreneur perspective
persona               # Check current mode
```

### 📊 **Charts & Visualization**
```bash
timeline                    # Static failure timeline
funding vs burn             # Static scatter plot
interactive dashboard       # Interactive scatter plot
interactive timeline        # Interactive timeline
sector dashboard           # Multi-chart dashboard
```

### 🎯 **Analysis Tools**
```bash
assess idea                 # 8-factor viability scoring
benchmark                   # Compare your idea to dataset
portfolio                   # Multi-startup heatmap analysis
insights                    # Auto-generated recommendations
```

### 🔍 **Web Scraping**
```bash
scrape <url>                # Scrape website paragraphs
scrape <url> <selector>     # Scrape with custom CSS selector

# Examples:
scrape https://techcrunch.com h1,h2    # Headlines
scrape https://startup.com .content    # By CSS class
scrape https://company.com #about      # By element ID
```

### 📄 **Reports & Export**
```bash
generate report             # Full portfolio PDF report
generate report for [name]  # Single startup PDF report
```

### 📁 **Data Management**
```bash
upload csv                 # Upload custom dataset
```

## 🎪 **Usage Examples**

### 💼 **Investor Workflow**
```bash
You: investor mode
NAVADA: 💼 **INVESTOR MODE** - VC perspective activated

You: portfolio
NAVADA: [Generates investment recommendation heatmap]

You: scrape https://competitor-startup.com
NAVADA: 🔵 **Website Analysis**
• Market opportunity: $2.5B TAM mentioned
• Revenue model: SaaS with enterprise focus
• Risk factors: Heavy tech dependency
```

### 🚀 **Founder Workflow**
```bash
You: founder mode
NAVADA: 🚀 **FOUNDER MODE** - Entrepreneur perspective activated

You: assess idea
NAVADA: [Walks through 8-factor viability assessment]

You: benchmark
NAVADA: [Compares your metrics against successful startups]

You: scrape https://successful-competitor.com
NAVADA: 🟢 **Website Analysis**
• Actionable insights for your execution strategy
• Tactical recommendations based on their approach
```

## 🏗️ **Project Structure**

```
Startup-Viability-Agent/
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (create this)
├── .chainlit/
│   └── config.toml            # Chainlit configuration
├── public/
│   └── how-to-use.html        # Comprehensive user guide
├── test_*.py                  # Test scripts
└── README.md                  # This file
```

## 🔧 **Configuration**

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
CHAINLIT_PORT=8000
CHAINLIT_HOST=localhost
```

### Chainlit Settings
- **Session timeout**: 1 hour
- **User session**: 15 days
- **File uploads**: Enabled (CSV, 500MB max)
- **Interactive mode**: Full CoT display

## 🔒 **Security Features**

### Web Scraping Safety
- **URL validation**: HTTP/HTTPS only
- **Content limits**: 5MB maximum
- **Request timeouts**: 15 seconds
- **Blocked domains**: localhost, file://, etc.
- **Content filtering**: Removes scripts, styles

### Data Protection
- **No sensitive data logging**
- **Environment variable encryption**
- **Session-based memory** (not persistent)
- **Secure API communication**

## 🧪 **Testing**

Run the comprehensive test suite:

```bash
# Test core features
python test_new_features.py

# Test advanced features
python test_advanced_features.py

# Test web scraping
python test_web_scraping.py

# Test portfolio functionality
python test_portfolio_fix.py
```

## 📊 **Built-in Dataset**

NAVADA includes a curated dataset of 12 startups across multiple sectors:

- **Sectors**: Tech, FinTech, HealthTech, EdTech, Transport, Energy, etc.
- **Metrics**: Funding, burn rate, founder experience, market size
- **Outcomes**: 50% success rate for realistic analysis
- **Geography**: US, UK, Germany, France

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Create a Pull Request**

## 🐛 **Troubleshooting**

### Common Issues

**Charts not generating?**
```bash
# Use exact keywords
timeline                    # ✅ Correct
show me timeline            # ❌ Won't work
```

**Web scraping fails?**
```bash
# Check URL in browser first
# Try different CSS selectors: h1, div, span
# Some sites block automated requests
```

**OpenAI API errors?**
```bash
# Check .env file exists and has valid API key
# Verify API key has sufficient credits
# Check internet connection
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Chainlit** - Amazing conversational AI framework
- **OpenAI** - GPT-4 integration for intelligent analysis
- **Plotly** - Interactive visualization capabilities
- **Community** - Feedback and feature suggestions

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/leeakpareva/Startup-Viability-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/leeakpareva/Startup-Viability-Agent/discussions)

---

## 🌟 **Star the Repository!**

If NAVADA helps with your startup analysis, please ⭐ this repository to help others discover it!

**Built with ❤️ for the startup community**