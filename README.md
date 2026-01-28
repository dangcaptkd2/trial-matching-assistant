# Clinical Trial Assistant Documentation

Welcome to the technical documentation for the Clinical Trial Assistant. This project is an AI-powered system designed to help patients and researchers find, understand, and compare clinical trials.

## Documentation Index

### 🏗️ [System Design](docs/system_design.md)
Explore the high-level architecture, including the User Interface, Backend API, LangGraph Agent Core, and Data Layer. Learn about the technology stack and how components interact.

### 🧠 [Workflow Logic](docs/workflow_logic.md)
Deep dive into the LangGraph agent. Visualizes the decision flow, state management, and the specific logic behind intent classification, reranking, and trial operations.

### 🔄 [Data Pipeline](docs/data_pipeline.md)
Understand how the system stays up-to-date with the latest data from ClinicalTrials.gov. This section details the automated download, import, transformation, and indexing processes.

### 📊 [Benchmarks & Evaluation](docs/benchmarks.md)
Review the methodologies used to validate the system's performance. Covers test suites for trial matching, eligibility checking, summarization, and more.

---

## Quick Links
- **[Docker Guide](DOCKER_GUIDE.md)**: Instructions for containerized deployment.
- **[API Source Code](src/api/)**: Direct link to the API implementation.
