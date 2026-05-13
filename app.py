# AI Travel Planner using LangGraph + Groq + Streamlit
import os
import streamlit as st

from typing import TypedDict, List, Dict, Annotated
from operator import add

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# =========================
# LOAD ENV & SETUP LLM
# =========================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# =========================
# STATE DEFINITION
# =========================
class TravelState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add]
    
    source_city: str
    destination: str
    budget: str
    days: int
    start_date: str
    end_date: str
    interests: List[str]
    hotel_type: str
    transport_mode: str

    itinerary: List[Dict]
    hotels: List[Dict]
    transport_options: List[Dict]
    places_to_visit: List[str]
    food_recommendations: List[str]
    
    estimated_cost: str  # Kept as string for formatted currency output
    current_step: str
    errors: List[str]

# =========================
# LANGGRAPH NODES
# =========================

def collect_preferences(state: TravelState):
    return {"current_step": "Preferences Collected"}

def find_places(state: TravelState):
    destination = state["destination"]
    interests = ", ".join(state["interests"])

    prompt = f"""
    Suggest the top 8 tourist attractions in {destination}.
    User interests: {interests}
    Give ONLY the place names, one per line. No extra text.
    """
    response = llm.invoke(prompt)
    places = [p.strip("- *") for p in response.content.split("\n") if p.strip()][:8]
    
    return {"places_to_visit": places, "current_step": "Places Found"}

def find_hotels(state: TravelState):
    prompt = f"""
    Suggest 4 {state['hotel_type']} hotels in {state['destination']}.
    Budget context: {state['budget']}
    Return ONLY in this exact format:
    Hotel Name - Price Per Night
    """
    response = llm.invoke(prompt)
    hotels = []
    for line in response.content.split("\n"):
        if "-" in line:
            parts = line.split("-")
            hotels.append({"name": parts[0].strip(), "price": parts[1].strip()})
            
    return {"hotels": hotels[:4], "current_step": "Hotels Found"}

def find_transport(state: TravelState):
    mode = state["transport_mode"]
    prompt = f"""
    Suggest 3 {mode} options from {state['source_city']} to {state['destination']}.
    Return ONLY in this exact format:
    Provider Name - Estimated Price
    """
    response = llm.invoke(prompt)
    options = []
    for line in response.content.split("\n"):
        if "-" in line:
            parts = line.split("-")
            options.append({"provider": parts[0].strip(), "price": parts[1].strip()})
            
    return {"transport_options": options[:3], "current_step": f"{mode} Options Found"}

def food_recommendation_node(state: TravelState):
    prompt = f"Suggest 6 famous foods/dishes in {state['destination']}. Return ONLY the food names, one per line."
    response = llm.invoke(prompt)
    foods = [f.strip("- *") for f in response.content.split("\n") if f.strip()][:6]
    
    return {"food_recommendations": foods, "current_step": "Foods Found"}

def generate_itinerary(state: TravelState):
    days = state["days"]
    places = ", ".join(state["places_to_visit"])
    foods = ", ".join(state["food_recommendations"])

    prompt = f"""
    Create a highly detailed, engaging {days}-day itinerary for {state['destination']}.
    Incorporate these places: {places}
    Incorporate these foods: {foods}
    
    Format the output using Markdown with headers for each day (e.g., ### Day 1).
    Include Morning, Afternoon, and Evening plans.
    """
    response = llm.invoke(prompt)
    
    return {"itinerary": [{"plan": response.content}], "current_step": "Itinerary Generated"}

def calculate_budget(state: TravelState):
    hotel_list = "\n".join([f"{h['name']}: {h['price']}" for h in state.get("hotels", [])])
    transport_list = "\n".join([f"{t['provider']}: {t['price']}" for t in state.get("transport_options", [])])

    prompt = f"""
    User's target budget: {state['budget']}.
    Trip duration: {state['days']} days.
    
    Found Prices:
    Hotels: {hotel_list}
    Transport: {transport_list}
    
    Calculate a realistic total estimated cost for the entire trip (including food & buffer).
    Return ONLY the final estimated amount as a string with the currency (e.g., "1,45,000 INR"). Do not explain your math.
    """
    response = llm.invoke(prompt)
    
    return {"estimated_cost": response.content.strip(), "current_step": "Budget Calculated"}

def final_response(state: TravelState):
    return {"current_step": "Completed"}

# =========================
# BUILD LANGGRAPH
# =========================
builder = StateGraph(TravelState)

builder.add_node("collect_preferences", collect_preferences)
builder.add_node("find_places", find_places)
builder.add_node("find_hotels", find_hotels)
builder.add_node("find_transport", find_transport)
builder.add_node("food_recommendation_node", food_recommendation_node)
builder.add_node("generate_itinerary", generate_itinerary)
builder.add_node("calculate_budget", calculate_budget)
builder.add_node("final_response", final_response)

builder.add_edge(START, "collect_preferences")
builder.add_edge("collect_preferences", "find_places")
builder.add_edge("find_places", "find_hotels")
builder.add_edge("find_hotels", "find_transport")
builder.add_edge("find_transport", "food_recommendation_node")
builder.add_edge("food_recommendation_node", "generate_itinerary")
builder.add_edge("generate_itinerary", "calculate_budget")
builder.add_edge("calculate_budget", "final_response")
builder.add_edge("final_response", END)

graph = builder.compile()


# =========================
# BEAUTIFUL STREAMLIT UI
# =========================
st.set_page_config(page_title="VoyageAI Planner", layout="wide", page_icon="✈️")

# Custom CSS for Premium Look (Theme-Friendly)
st.markdown("""
    <style>
    /* Hide top header */
    header { visibility: hidden; }
    
    /* Make Metric Values Pop */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #60A5FA; /* A nice bright blue that works well in dark mode */
    }
    
    /* Primary Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2060/2060284.png", width=60)
    st.title("VoyageAI")
    st.caption("Your intelligent travel architect.")
    st.divider()
    
    st.subheader("📍 Journey Details")
    source_city = st.text_input("From", "Mumbai", placeholder="E.g., Delhi")
    destination = st.text_input("To", "Kyoto", placeholder="E.g., Paris")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
        
    days = (end_date - start_date).days
    days = days if days > 0 else 1 # Fallback if same day
    
    st.divider()
    st.subheader("💰 Preferences")
    budget = st.text_input("Total Budget", "150,000 INR")
    
    transport_mode = st.selectbox("Travel Mode", ["Flight", "Train", "Bus"])
    hotel_type = st.select_slider("Accommodation Style", options=["Budget", "3-star", "4-star", "5-star", "Luxury"])
    
    interests = st.multiselect(
        "What excites you?", 
        ["Food", "Adventure", "History", "Nature", "Shopping", "Art", "Nightlife"],
        default=["Food", "History", "Nature"]
    )
    
    st.write("")
    generate_btn = st.button("✨ Generate Master Plan")


# --- MAIN DASHBOARD AREA ---
if not generate_btn:
    # Landing Page State
    st.title("🌍 Plan Your Dream Trip with AI")
    st.markdown("Fill out your preferences in the **sidebar** and click **Generate Master Plan** to let our AI agents negotiate the best routes, stays, and activities for you.")
    
    st.image(
        "https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=1200&q=80", 
        use_container_width=True, 
        caption="Let's build your next adventure."
    )

else:
    # Loading State
    with st.spinner(f"🤖 AI Agents are orchestrating your {days}-day trip to {destination}..."):
        
        # Execute Graph
        result = graph.invoke({
            "messages": [],
            "source_city": source_city,
            "destination": destination,
            "budget": budget,
            "days": days,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "interests": interests,
            "hotel_type": hotel_type,
            "transport_mode": transport_mode,
            "itinerary": [],
            "hotels": [],
            "transport_options": [],
            "places_to_visit": [],
            "food_recommendations": [],
            "estimated_cost": "",
            "current_step": "",
            "errors": []
        })

    # --- RESULTS DASHBOARD ---
    st.title(f"✈️ Your {days}-Day Escape to {destination}")
    
    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Destination", destination)
    m2.metric("Duration", f"{days} Days")
    m3.metric("Transport", transport_mode)
    m4.metric("Est. Total Cost", result["estimated_cost"])
    
    st.write("")
    
    # Organized Tabs
    tab1, tab2, tab3 = st.tabs(["🗓️ The Itinerary", "🛌 Logistics & Booking", "📸 Highlights"])
    
    with tab1:
        st.subheader("Your Daily Schedule")
        with st.container(border=True):
            st.markdown(result["itinerary"][0]["plan"])
            
    with tab2:
        col_t, col_h = st.columns(2)
        
        with col_t:
            st.subheader(f"🎫 {transport_mode} Options")
            if result["transport_options"]:
                for t in result["transport_options"]:
                    with st.container(border=True):
                        st.markdown(f"**{t.get('provider', 'Provider')}**")
                        st.caption(f"Estimated: {t.get('price', 'N/A')}")
            else:
                st.info("No transport options found.")
                
        with col_h:
            st.subheader(f"🏨 {hotel_type} Stays")
            if result["hotels"]:
                for h in result["hotels"]:
                    with st.container(border=True):
                        st.markdown(f"**{h.get('name', 'Hotel')}**")
                        st.caption(f"Nightly Rate: {h.get('price', 'N/A')}")
            else:
                st.info("No hotel options found.")
                
    with tab3:
        col_p, col_f = st.columns(2)
        
        with col_p:
            st.subheader("📍 Must-Visit Attractions")
            for place in result["places_to_visit"]:
                st.markdown(f"- {place}")
                
        with col_f:
            st.subheader("🍜 Culinary Journey")
            for food in result["food_recommendations"]:
                st.markdown(f"- {food}")

    st.toast("Trip successfully generated!", icon="🎉")