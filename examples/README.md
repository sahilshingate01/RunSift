# RunSift Agent Adapters (Untested Examples)

> **Note:** The framework integration scripts in this directory (`langgraph_agent.py`, `crewai_researcher.py`, and `autogen_coding_assistant.py`) are **untested examples** provided for reference. Measured benchmarks are in `benchmarks/tensorroute_benchmark/`.

This directory contains integration examples showing how OpenAI-compatible agent frameworks can point their `base_url` at `http://127.0.0.1:8765/v1`:
1. **Graph-Based Agents** (LangGraph — untested example)
2. **Role & Process-Based Multi-Agent Crews** (CrewAI — untested example)
3. **Conversational Ping-Pong Loops** (Microsoft AutoGen — untested example)

---

## 1. Quick Start: Closed-Loop Demo with Live Dashboard

### Step 1: Start the RunSift Gateway
```bash
uvicorn src.proxy.gateway:app --host 127.0.0.1 --port 8765
```

### Step 2: Open the Real-Time Dashboard
Open your browser to:
**[http://127.0.0.1:8765/dashboard](http://127.0.0.1:8765/dashboard)**

### Step 3: Run Any Agent Adapter
In a second terminal, execute any of the following adapters:

#### A. Lightweight Multi-Turn Agent (Zero Third-Party Dependencies)
```bash
python examples/run_agent_traffic.py
```

#### B. LangGraph ReAct Autonomous Agent (Untested Example)
```bash
pip install langgraph langchain-openai langchain-core
python examples/langgraph_agent.py
```

#### C. CrewAI Multi-Agent Research & Coding Crew (Untested Example)
```bash
pip install crewai
python examples/crewai_researcher.py
```

#### D. AutoGen ConversableAgent Coding Assistant (Untested Example)
```bash
pip install autogen-agentchat~=0.2 pyautogen
python examples/autogen_coding_assistant.py
```

---

## 2. Gateway Features Exercised

1. **Prefix Canonicalization:**
   - Reorganizes tool schema keys, sorts function parameter definitions, and hoists volatile session timestamps behind static prompt instructions.

2. **Tool-Output Compaction:**
   - Condenses verbose tool output before forwarding requests to the upstream OpenAI-compatible endpoint.

