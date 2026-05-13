# 🌍 VoyageAI: Smart AI Travel Planner

VoyageAI is an intelligent, agentic travel planning application built using **LangGraph**, **Groq**, and **Streamlit**. It takes your travel preferences—like budget, interests, and duration—and orchestrates multiple AI nodes to negotiate the best routes, accommodations, and activities, culminating in a comprehensive master itinerary.

## ✨ Features
- **Dynamic Transport Options:** Automatically fetches flight, train, or bus options based on user preference.
- **Smart Budget Estimation:** Analyzes suggested transport and hotel prices to calculate a realistic total trip cost.
- **Agentic Workflow:** Utilizes LangGraph to break down the planning process into specialized, sequential AI tasks.
- **Beautiful UI:** A premium, dark-mode friendly Streamlit dashboard with tabs and metric cards.

---

## 🏗️ Architecture & Node Flow (UML Diagram)

The application uses **LangGraph** to manage the state of the travel plan. The workflow starts by collecting preferences and sequentially passes the state through various specialized nodes until the final plan is generated.

```mermaid
graph TD
    %% Define Styles
    classDef startEnd fill:#2563EB,stroke:#1D4ED8,stroke-width:2px,color:#fff,font-weight:bold;
    classDef node fill:#1F2937,stroke:#374151,stroke-width:2px,color:#60A5FA,font-weight:bold;

    %% Nodes
    START((START)):::startEnd
    N1[Collect Preferences]:::node
    N2[Find Places]:::node
    N3[Find Hotels]:::node
    N4[Find Transport]:::node
    N5[Food Recommendations]:::node
    N6[Generate Itinerary]:::node
    N7[Calculate Budget]:::node
    N8[Final Response]:::node
    END((END)):::startEnd

    %% Edges
    START --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> END
