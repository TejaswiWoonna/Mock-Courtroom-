# ⚖️ AI Courtroom Simulator

A **multi-agent, AI-powered virtual courtroom system** built using **CrewAI** that simulates real-world legal trials through autonomous agents representing courtroom roles such as **Prosecution, Defense, Judge, Jury, Witnesses, Case Loader, and Reporter**.

This project demonstrates advanced **agent orchestration, legal reasoning, argument generation, and decision-making**, fulfilling academic requirements for **context sharing, tool integration, structured output, monitoring, and agent-to-agent collaboration**.

---

## 🎯 Project Objectives

* Simulate realistic courtroom proceedings using AI agents
* Enable dynamic legal argument generation (10+ interactions)
* Demonstrate multi-agent communication and orchestration
* Apply Retrieval-Augmented Generation (RAG) for case understanding
* Generate structured verdicts and trial summaries

---

## 🧠 System Architecture

Each courtroom role is implemented as an independent **CrewAI Crew**, consisting of agents, tasks, and tools.

### 🧩 Crews Overview

| Crew            | Responsibility                       |
| --------------- | ------------------------------------ |
| CaseLoaderCrew  | Loads and parses legal case datasets |
| ProsecutionCrew | Generates prosecution arguments      |
| DefenseCrew     | Generates defense counterarguments   |
| WitnessCrew     | Produces witness testimonies         |
| JuryCrew        | Evaluates arguments impartially      |
| JudgeCrew       | Issues final verdict                 |
| ReporterCrew    | Generates full trial report          |

---

## 📁 Project Structure

```bash
src/
 └── courtroom/
      ├── crews/
      │     ├── case_loader_crew/
      │     ├── prosecution_crew/
      │     ├── defense_crew/
      │     ├── witness_crew/
      │     ├── jury_crew/
      │     ├── judge_crew/
      │     └── reporter_crew/
      ├── tools/
      │     ├── file_parser_tool.py
      │     ├── evidence_analyzer_tool.py
      │     └── a2a_protocol.py
      │     └──legal_search_tool.py
      ├── data/
      │     └── legal_cases.json
      └── main.py
```

---

## 🛠️ Technologies Used

* **Python 3.10+**
* **CrewAI** (Multi-Agent Framework)
* **LLMs**: Gemini / Groq / OpenAI
* **Pydantic** (Structured Output)
* **YAML** (Agent & Task Configuration)
* **RAG (Retrieval-Augmented Generation)**

---

## 🔁 Trial Execution Flow

1. Load legal case data
2. Prosecution presents arguments
3. Defense generates counterarguments
4. Witnesses provide testimony
5. Jury evaluates both sides
6. Judge delivers verdict
7. Reporter generates trial summary

Each step passes context using **shared state** across agents.

---

## 📊 Minimum Project Criteria Mapping

| Requirement                  | Status                       |
| ---------------------------- | ---------------------------- |
| Context Sharing              | ✅ Implemented via Flow state |
| Tool Integration (MCP)       | ✅ Custom tools used          |
| Structured Output            | ✅ Pydantic models            |
| Logging & Monitoring         | ✅ Callback logging           |
| Agent-to-Agent Communication | ✅ CrewAI interoperability    |
| Framework Usage              | ✅ CrewAI                     |

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export GROQ_API_KEY=your_key
export GEMINI_API_KEY=your_key
```

---

## ▶️ Run the Simulation

```bash
python src/courtroom/main.py
```

---

## 📄 Sample Output

* Judge Verdict (JSON)
* Trial Transcript (Markdown)
* Lawyer Argument Logs

---


