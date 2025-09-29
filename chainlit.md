# 🚀 NAVADA - Startup Viability Agent

**AI-Powered Analysis with Voice Support & Real-Time Intelligence**

Welcome to NAVADA, your intelligent startup analysis companion that combines AI expertise with real-time market data and voice interaction.

---

## 🎯 Quick Start (30 Seconds)

### Step 1: Your First Commands
Try these instantly:
- Type: `timeline`
- Type: `assess idea`
- Type: `Show me a bar chart of funding by sector`

### Step 2: Understanding NAVADA
**NAVADA** = **N**ew **A**nalysis & **V**iability **A**ssessment for **D**ata-driven **A**nalysis

NAVADA is your AI-powered startup analyst that:
- ✅ Generates visual charts automatically
- ✅ Analyzes startup risk factors
- ✅ Predicts failure patterns
- ✅ Scores startup viability (0-100)
- ✅ Searches real-time market data
- ✅ Provides voice responses

---

## 🎭 Choose Your Mode

### 💼 **Investor Mode**
Focus on ROI, market analysis, and investment opportunities
```
investor mode
```

### 🚀 **Founder Mode**
Focus on execution, operations, and tactical recommendations
```
founder mode
```

---

## 🎯 Three Ways to Use NAVADA

### 1️⃣ **Predefined Chart Commands**
Type exact keywords for instant charts:

| Command | What You Get |
|---------|-------------|
| `timeline` | Bar chart showing when each startup will run out of money |
| `funding vs burn` | Scatter plot showing funding amount vs burn rate (colored by success/failure) |
| `sector comparison chart` | Bar chart comparing average funding across industries |
| `failure rate by country` | Color-coded bars showing failure % for UK, US, Germany, France |
| `experience vs success chart` | Scatter plot showing founder experience vs outcome |

**Example:**
```
You: timeline
NAVADA: [Generates failure timeline chart with 12 startups]
```

### 2️⃣ **Natural Language Chart Requests**
Ask for any chart in plain English:

**Examples:**
```
You: Show me a bar chart of funding by startup
NAVADA: [AI detects intent, generates custom bar chart]

You: Create a pie chart of successful vs failed companies
NAVADA: [Generates pie chart with proportions]

You: Visualize the relationship between market size and funding
NAVADA: [Creates scatter plot]
```

**Supported Chart Types:**
- `bar chart` - Compare categories
- `scatter plot` - Show relationships
- `line chart` - Display trends
- `pie chart` - Show proportions

### 3️⃣ **Ask Questions (AI Analysis)**
Type natural questions about the data:

**Sample Questions:**
```
You: Which sector looks riskiest?
NAVADA: Based on the data, Transport has a 100% failure rate...

You: What's the average funding for successful startups?
NAVADA: Successful startups have an average funding of $11.3M...

You: Why do some startups fail faster than others?
NAVADA: [Provides detailed analysis based on burn rate, funding, experience factors]
```

---

## 🔍 **Real-Time Market Intelligence**
```
search AI startups 2024
latest trends in fintech
what's happening with blockchain
current venture capital news
```

## 🎤 **Voice Control**
```
voice on          # Enable text-to-speech
voice off         # Disable voice output
```

## 🧮 **Mathematical Analysis Mode**

### How to Enter Math Mode:
```
math mode
```

### Advanced Calculations Available:

#### 💰 **IRR & Investment Returns**
```
calculate IRR for 5x return in 7 years
```
**Example Output:** 5x return in 7 years = **25.9% annual return**

#### 📈 **Revenue Projections**
```
project revenue with 20% monthly growth
```
**Shows:** Starting MRR → Final MRR → Total Growth % → ARR projections

#### 🎯 **Monte Carlo Simulations**
```
simulate 1000 scenarios for exit
```
**Provides:** Mean/Median exit values, percentiles, best/worst cases

#### 🔥 **Burn Rate Optimization**
```
optimize burn rate for 18 month runway
```
**Calculates:** Optimal monthly/annual burn rates for desired runway

### Exit Math Mode:
```
exit math mode
```

## 📊 **Analytics & Charts**
```
show funding vs burn chart
display growth trajectory
generate portfolio analysis
create risk assessment
```

---

## 🎯 Viability Assessment (Interactive Scoring)

### How to Start:
```
You: assess idea
```

### What Happens:
1. **NAVADA asks 9 questions** about your startup:
   - Funding amount (USD Millions)
   - Burn rate (months)
   - Team experience (years)
   - Market size (Billions)
   - Business model strength (1-5)
   - Moat/defensibility (1-5)
   - Current MRR ($k)
   - Monthly growth rate (%)
   - Competition intensity (1-5)

2. **You provide answers** (or press Enter for defaults)

3. **NAVADA generates:**
   - Visual gauge showing score (0-100)
   - Score interpretation: 🟢 Strong | 🟡 Moderate | 🔴 Weak
   - Estimated runway (months)
   - Projected failure year
   - Component breakdown (8 factors)
   - Actionable recommendations

---

## 📄 **Generate Reports**
```
generate report
portfolio analysis
```

---

## 🔧 Features

- **24 Startup Dataset** with enhanced metrics
- **6 Chart Types** for comprehensive visualization
- **Mathematical Analysis Mode** with IRR, NPV, Monte Carlo simulations
- **LangSmith Thread Tracking** for conversation continuity
- **Brave Search Integration** for real-time market data
- **OpenAI TTS** for voice responses
- **PDF Report Generation** for professional presentations

---

## 💡 Example Conversations

**For Investors:**
- "Show me the portfolio heatmap"
- "Search latest SaaS funding rounds"
- "Generate investment report for TechFlow"

**For Founders:**
- "Assess my startup idea"
- "What's the current state of fintech competition?"
- "Show growth trajectory analysis"

**For Financial Analysis:**
- "Enter math mode and calculate 10x return in 5 years"
- "Project our growth to $1M ARR with current rates"
- "Simulate 500 exit scenarios for our sector"

---

## 📁 Upload Your Own Data

### How to Upload:
```
You: upload csv
NAVADA: [Prompts file upload]
```

### Required Columns:
| Column | Description | Example |
|--------|-------------|---------|
| `Startup` | Company name | "TechCorp" |
| `Funding_USD_M` | Funding in millions | 5.5 |
| `Burn_Rate_Months` | Burn rate | 12 |
| `Failed` | 0=success, 1=failed | 1 |
| `Country` | Country code | "UK" |
| `Sector` | Industry | "HealthTech" |

---

## 🐛 Troubleshooting

### Chart Not Generating?
**Solutions:**
1. Use exact keywords: `timeline` not "show timeline"
2. Be explicit: "Create a BAR CHART of X"
3. Try predefined command first: `sector comparison chart`

**Example:**
```
❌ You: timeline please
✅ You: timeline

❌ You: I want to see funding
✅ You: Show me a bar chart of funding by startup
```

### AI Not Understanding Question?
**Solutions:**
1. Be more specific
2. Use column names
3. Break into smaller questions

**Example:**
```
❌ You: What's the best?
✅ You: Which startup has the highest funding?

❌ You: Show me stuff
✅ You: Show me a bar chart of Funding_USD_M by Sector
```

---

## 💡 Pro Tips & Tricks

### Tip 1: Chain Commands
```
You: timeline
[Review chart]
You: Now show me the riskiest sectors
[Get analysis]
You: sector comparison chart
[See visualization]
```

### Tip 2: Use Follow-up Questions
```
You: Which country has highest failure rate?
NAVADA: UK has 71.4% failure rate (5 out of 7).
You: Why?
NAVADA: [Provides detailed analysis]
You: Show me a chart
NAVADA: [Generates relevant visualization]
```

### Tip 3: Combine Data + Charts
```
You: What's the average funding for failed startups?
NAVADA: $2.9M average for failed startups.
You: Compare that visually with successful startups
NAVADA: [Generates comparison chart]
```

---

## 📊 Available Data Columns

Use these in your natural language requests:

| Column | Description | Use In Requests |
|--------|-------------|----------------|
| `Startup` | Company name | "by Startup", "for each startup" |
| `Funding_USD_M` | Funding (millions) | "funding amount", "investment" |
| `Burn_Rate_Months` | Burn rate | "burn rate", "runway" |
| `Founders_Experience_Yrs` | Founder experience | "experience", "founder years" |
| `Market_Size_Bn` | Market size (billions) | "market size", "TAM" |
| `Failed` | Success/Failure | "outcome", "success", "failed" |
| `Country` | Country code | "by country", "geographic" |
| `Sector` | Industry | "by sector", "industry", "vertical" |
| `Est_Failure_Year` | Failure projection | "failure year", "when they fail" |

---

## 🎓 Success Metrics

**You're using NAVADA effectively if:**
- ✅ You can generate charts in < 10 seconds
- ✅ You understand your data patterns
- ✅ You get actionable recommendations
- ✅ You can assess startup viability
- ✅ You make data-driven decisions

---

## 🆘 Need Help?

### Common Commands:
- `timeline` - Quick test
- `assess idea` - Interactive guidance
- `math mode` - Enter mathematical analysis mode
- `search [topic]` - Real-time market data
- `voice on/off` - Toggle voice responses

### Ask NAVADA:
```
You: How do I create charts?
You: What commands are available?
You: Show me an example
```

---

**Ready to start analyzing? Just type your question below! 👇**

**Designed and Developed by Lee Akpareva MBA, MA**