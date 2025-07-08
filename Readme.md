# Multi-Agent AI System

## Overview

<p>The Root Agent System is a unified multi-agent framework built using LangChain and Gemini 2.0 Flash. It integrates six task-specific agents under one root controller. Each agent performs a specialized function such as research, data analysis, automation, conversational Q&A, sentiment analysis, and decision-making support.<p>

### Features

- Information Retrieval Agent – Summarizes content from the web, Wikipedia, or uploaded PDFs.

- Data Analysis Agent – Performs insights and calculations on structured datasets (CSV, JSON).

- Task Automation Agent – Automates workflows such as email drafting and report generation.

- Conversational Q&A Agent – Maintains context-aware dialogue across multiple user inputs.

- Decision Support Agent – Compares options using structured logic or external data.

- Sentiment Analysis Agent – Analyzes textual data to classify sentiment or tone.

### Technologies Used

- LangChain (agents, tools, memory)

- Gemini 2.0 Flash (via langchain_google_genai)

- Tavily, Wikipedia, or DuckDuckGo search

- Matplotlib, Pandas

- Python (CLI interface)

- etc

---

# Working Steps

### Environment Setup

Loads .env configuration using dotenv, containing required API keys.

### Memory Initialization

Uses ConversationBufferMemory to preserve context during chat sessions.

### Prompt Loading

Loads a dynamic prompt (react-chat) from LangChain Hub.

### LLM Configuration

Initializes ChatGoogleGenerativeAI with Gemini 2.0 Flash.

### Agent Construction

Uses create_react_agent() to build a reasoning-based agent that selects the appropriate tool.

### Agent Execution

AgentExecutor handles execution, parsing, and tool dispatch with memory support.

### User Interaction

A CLI loop continuously accepts user input, invokes the appropriate tool, and displays the result.

## Sample Queries

- Summarize recent advancements in quantum computing

- Analyze customer reviews in this dataset

- What were Q1 sales for product X?

- Draft an email to the manager about task completion

- Compare two phones based on battery life and cost

---

---

# Agents

## Agent 1 :

Information Retrieval Agent (Research Assistant)

Task: Build an agent that retrieves and summarizes information from the web or a provided knowledge base (e.g., Wikipedia, PDFs, or web search).

Features:

- Uses LangChain’s web search tool (e.g., Tavily, SerpAPI) or document loader.
- Summarizes findings in concise, user-friendly responses.
- Example query: “Summarize recent advancements in quantum computing.”
- Skills Learned: Web scraping, document loading, summarization, tool integration.
- Tools: LangChain’s WebBaseLoader, TavilySearchResults, or Wikipedia loader.

## Agent 2: Data Analysis Agent

Task: Build an agent that processes and analyzes structured data (e.g., CSV, JSON) and answers queries based on it.

Features:

- Loads and parses structured data (e.g., sales data, weather data).
- Performs calculations or generates insights (e.g., “What’s the average sales for Q1?”).
- Optional: Visualize results using a simple chart (e.g., matplotlib integration).
- Skills Learned: Data handling, tool creation, integration with external libraries.
- Tools: LangChain’s PandasDataFrameLoader, custom tools for calculations

## Agent 3: Task Automation Agent

Task: Build an agent that automates a specific workflow, such as drafting emails, scheduling tasks, or generating reports.

Features:

- Takes user inputs to generate structured outputs (e.g., “Draft an email to a client about a project update”).
- Integrates with APIs (e.g., email or calendar APIs, if feasible).
- Example query: “Schedule a meeting for next Monday at 10 AM.”
- Skills Learned: API integration, prompt engineering, automation workflows.
- Tools: LangChain’s Tool class, external APIs (e.g., Gmail, Google Calendar).

## Agent 4:Conversational Q&A Agent with Memory

Task: Build an agent that handles conversational queries with context retention, acting as a domain-specific chatbot (e.g., tech support, HR assistant).

Features:

- Maintains conversation history using LangChain’s memory (e.g., ConversationBufferMemory).
- Answers follow-up questions with context (e.g., “Can you clarify what you meant earlier?”).
- Example query: “Explain neural networks, then give me a simpler explanation.”
- Skills Learned: Memory management, conversational AI, prompt chaining.
- Tools: LangChain’s ConversationChain, ConversationBufferMemory.

## Agent 5: Decision Support Agent

Task: Build an agent that provides decision-making support by evaluating options based on user-defined criteria or a predefined framework.

Features:

- Takes user inputs specifying a decision problem (e.g., “Should I invest in stock A or B?” or “Which marketing strategy is better for product X?”).
- Uses a decision-making framework (e.g., weighted scoring, pros/cons analysis) to evaluate options.
- Optionally integrates external data (e.g., stock prices, market trends) using LangChain tools.
- Returns a clear recommendation with reasoning (e.g., “Stock A is recommended due to higher ROI and lower risk.”).
- Example query: “Compare two marketing campaigns based on cost and expected reach.”
- Skills Learned: Decision-making logic, custom tool creation, prompt engineering, integration with external data sources.
- Tools: LangChain’s Tool class for custom decision logic, optional integration with APIs (e.g., Yahoo Finance for stock data or mock data for simplicity)

## Agent 6: Sentiment Analysis Agent

Task: Build an agent that analyzes text to determine sentiment, tone, or emotional context (e.g., customer reviews, social media posts).

Features:

- Processes text inputs to classify sentiment (e.g., positive, negative, neutral) or detect emotions (e.g., happy, angry).
- Provides summaries or insights (e.g., “80% of reviews are positive, with common themes of product quality”).
- Example query: “Analyze customer feedback on product X from this dataset.”
- Optionally integrates with external sources (e.g., X posts via API or mock data).
- Skills Learned: Natural language processing, sentiment analysis, custom tool creation, text summarization.
- Tools: LangChain’s Tool class for sentiment analysis (e.g., using Hugging Face’s sentiment models or NLTK), optional XPostLoader for social media data.
