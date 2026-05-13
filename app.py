# AI Travel Planner using LangGraph + Groq + Streamlit
import os
import streamlit as st
from typing import TypedDict, List, Dict, Annotated
from operator import add
from dotenv import load_dotenv

# Load Environment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# =========================
# LLM & GRAPH LOGIC
# =========================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

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
    places_to_visit: List[Dict]  
    food_recommendations: List[Dict] 
    
    estimated_cost: str
    current_step: str

# --- Nodes ---

def collect_preferences(state: TravelState):
    return {"current_step": "Preferences Collected"}

def find_places(state: TravelState):
    prompt = f"Suggest the top 6 tourist attractions in {state['destination']} for interests: {', '.join(state['interests'])}. Give ONLY the place names, one per line."
    response = llm.invoke(prompt)
    
    place_names = [p.strip("- *") for p in response.content.split("\n") if p.strip()][:6]
    
    places_with_images = []
    safe_dest = state['destination'].replace(' ', ',')
    for name in place_names:
        safe_name = name.replace(' ', ',')
        image_url = f"https://loremflickr.com/600/400/{safe_name},{safe_dest}/all"
        places_with_images.append({
            "name": name,
            "image": image_url
        })
        
    return {"places_to_visit": places_with_images, "current_step": "Places Found"}

def find_hotels(state: TravelState):
    prompt = f"Suggest 4 {state['hotel_type']} hotels in {state['destination']} for budget {state['budget']}. Format: Name - Price"
    response = llm.invoke(prompt)
    hotels = [{"name": l.split("-")[0].strip(), "price": l.split("-")[1].strip()} for l in response.content.split("\n") if "-" in l][:4]
    return {"hotels": hotels, "current_step": "Hotels Found"}

def find_transport(state: TravelState):
    prompt = f"Suggest 3 {state['transport_mode']} options from {state['source_city']} to {state['destination']}. Format: Provider - Price"
    response = llm.invoke(prompt)
    options = [{"provider": l.split("-")[0].strip(), "price": l.split("-")[1].strip()} for l in response.content.split("\n") if "-" in l][:3]
    return {"transport_options": options, "current_step": f"{state['transport_mode']} Options Found"}

# --- UPDATED FOOD NODE ---
def food_recommendation_node(state: TravelState):
    prompt = f"Suggest 6 famous foods in {state['destination']}. Only names."
    response = llm.invoke(prompt)
    
    food_names = [f.strip("- *") for f in response.content.split("\n") if f.strip()][:6]
    
    foods_with_images = []
    for name in food_names:
        safe_name = name.replace(' ', ',')
        # We add "food" to the URL to ensure the placeholder grabs culinary images
        image_url = f"https://loremflickr.com/600/400/{safe_name},food/all"
        foods_with_images.append({
            "name": name,
            "image": image_url
        })
        
    return {"food_recommendations": foods_with_images, "current_step": "Food Found"}

def generate_itinerary(state: TravelState):
    place_names = [p["name"] for p in state.get("places_to_visit", [])]
    food_names = [f["name"] for f in state.get("food_recommendations", [])] 
    
    prompt = f"""Detailed {state['days']}-day itinerary for {state['destination']} 
    visiting: {', '.join(place_names)}
    and eating: {', '.join(food_names)}. 
    Use Markdown with headers for each day."""
    
    response = llm.invoke(prompt)
    return {"itinerary": [{"plan": response.content}], "current_step": "Itinerary Generated"}

def calculate_budget(state: TravelState):
    hotel_list = "\n".join([f"{h['name']}: {h['price']}" for h in state.get("hotels", [])])
    transport_list = "\n".join([f"{t['provider']}: {t['price']}" for t in state.get("transport_options", [])])

    prompt = f"""
    Target budget: {state['budget']}. Days: {state['days']}.
    Found Prices: 
    Hotels: {hotel_list}
    Transport: {transport_list}
    Calculate realistic total estimated cost. Return ONLY the final estimated amount and currency (e.g., '1,45,000 INR'). Do not explain.
    """
    response = llm.invoke(prompt)
    return {"estimated_cost": response.content.strip(), "current_step": "Completed"}

# --- Graph Build ---
builder = StateGraph(TravelState)

builder.add_node("collect_preferences", collect_preferences)
builder.add_node("find_places", find_places)
builder.add_node("find_hotels", find_hotels)
builder.add_node("find_transport", find_transport)
builder.add_node("food_recommendation_node", food_recommendation_node)
builder.add_node("generate_itinerary", generate_itinerary)
builder.add_node("calculate_budget", calculate_budget)

builder.add_edge(START, "collect_preferences")
builder.add_edge("collect_preferences", "find_places")
builder.add_edge("find_places", "find_hotels")
builder.add_edge("find_hotels", "find_transport")
builder.add_edge("find_transport", "food_recommendation_node")
builder.add_edge("food_recommendation_node", "generate_itinerary")
builder.add_edge("generate_itinerary", "calculate_budget")
builder.add_edge("calculate_budget", END)

graph = builder.compile()

# =========================
# PROFESSIONAL CHATBOT UI
# =========================
st.set_page_config(
    page_title="VoyageAI | Agent", 
    layout="wide", 
    page_icon="🤖", 
    initial_sidebar_state="expanded" 
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .main-header { font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #58A6FF, #BC8CFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    .chat-bubble { background-color: #1F2937; padding: 20px; border-radius: 15px; border-left: 5px solid #2563EB; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    .metric-card { background: #161B22; border: 1px solid #30363D; border-radius: 10px; padding: 15px; text-align: center; }
    .stButton>button { border-radius: 8px; background: linear-gradient(135deg, #238636 0%, #2ea043 100%); color: white; border: none; font-weight: 600; padding: 0.6rem; transition: transform 0.2s ease;}
    .stButton>button:hover { transform: translateY(-2px); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px 8px 0 0; padding: 10px 20px; color: #8B949E; }
    .stTabs [aria-selected="true"] { background-color: #1F2937; border-bottom: 2px solid #58A6FF; color: #58A6FF; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color: #58A6FF;'>🤖 Voyage Agent</h2>", unsafe_allow_html=True)
    st.caption("AI-Powered Travel Intelligence")
    st.divider()
    
    with st.expander("📍 Route Details", expanded=True):
        src = st.text_input("Origin", "Mumbai")
        dest = st.text_input("Destination", "Tokyo")
        col1, col2 = st.columns(2)
        s_date = col1.date_input("Start")
        e_date = col2.date_input("End")
    
    with st.expander("💳 Budget & Comfort", expanded=True):
        budget_val = st.text_input("Target Budget", "200,000 INR")
        trans_mode = st.selectbox("Transport", ["Flight", "Train", "Bus"])
        h_type = st.selectbox("Hotel Class", ["Budget", "3-star", "4-star", "5-star"])
    
    user_interests = st.multiselect("Interests", ["Food", "Adventure", "Culture", "Shopping", "Nature"], default=["Food", "Culture"])
    
    st.write("")
    generate_btn = st.button("🚀 Plan My Journey")

# --- Main Interface ---
if not generate_btn:
    st.markdown("<h1 class='main-header'>Welcome to VoyageAI</h1>", unsafe_allow_html=True)
    st.markdown("#### Your personal AI travel concierge. Fill in the details on the left to generate a professional travel blueprint.")
    
    st.write("")
    st.write("")
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.info("🧠 **Agentic Search**\n\nAI agents find real locations, activities, and hotels based on your specific interests.")
    with c2: 
        st.info("🚆 **Dynamic Transit**\n\nWhether you prefer high-speed trains or direct flights, the AI negotiates the best paths.")
    with c3: 
        st.info("📊 **Budget Optimized**\n\nSmart mathematical routing ensures your entire itinerary stays perfectly within range.")
        
else:
    with st.spinner("🤖 **Agent is formulating your perfect trip...**"):
        d_count = (e_date - s_date).days if (e_date - s_date).days > 0 else 1
        result = graph.invoke({
            "source_city": src, "destination": dest, "budget": budget_val,
            "days": d_count, "start_date": str(s_date), "end_date": str(e_date),
            "interests": user_interests, "hotel_type": h_type, "transport_mode": trans_mode,
            "itinerary": [], "hotels": [], "transport_options": [], "places_to_visit": [],
            "food_recommendations": [], "estimated_cost": "", "messages": []
        })

    # Summary Bar
    st.markdown(f"### 🌏 Journey to {dest}")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f"<div class='metric-card'><small>DURATION</small><br><b style='font-size: 1.2rem; color: #58A6FF;'>{d_count} Days</b></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='metric-card'><small>TRANSPORT</small><br><b style='font-size: 1.2rem; color: #58A6FF;'>{trans_mode}</b></div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='metric-card'><small>HOTEL STYLE</small><br><b style='font-size: 1.2rem; color: #58A6FF;'>{h_type}</b></div>", unsafe_allow_html=True)
    with m4: st.markdown(f"<div class='metric-card'><small>EST. TOTAL COST</small><br><b style='font-size: 1.2rem; color: #3FB950;'>{result['estimated_cost']}</b></div>", unsafe_allow_html=True)

    st.write("")

    # Chat-Style Results Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Full Itinerary", "🏨 Logistics & Booking", "✨ Highlights Gallery"])

    with tab1:
        with st.chat_message("assistant"):
            st.markdown("Here is your detailed daily schedule optimized for your interests:")
            st.markdown(f"<div class='chat-bubble'>{result['itinerary'][0]['plan']}</div>", unsafe_allow_html=True)

    with tab2:
        with st.chat_message("assistant"):
            st.write("I've found these top-rated options for your stay and travel:")
            c_h, c_t = st.columns(2)
            with c_h:
                st.subheader("🏠 Stays")
                for h in result["hotels"]:
                    st.success(f"**{h['name']}**\n\nEstimated Price: {h['price']}")
            with c_t:
                st.subheader(f"🎫 {trans_mode} Options")
                for t in result["transport_options"]:
                    st.warning(f"**{t['provider']}**\n\nEstimated Fare: {t['price']}")

    # --- UPDATED HIGHLIGHTS TAB WITH FOOD IMAGES ---
    with tab3:
        with st.chat_message("assistant"):
            st.write("Visualizing your journey... here are the top spots and eats you shouldn't miss:")
            
            st.subheader("📍 Must-Visit Attractions")
            cols = st.columns(2)
            for idx, place in enumerate(result.get("places_to_visit", [])):
                with cols[idx % 2]:
                    with st.container(border=True):
                        st.image(place['image'], use_container_width=True)
                        st.markdown(f"<h5 style='text-align: center;'>{place['name']}</h5>", unsafe_allow_html=True)

            st.divider()
            
            st.subheader("🍱 Local Eats")
            f_cols = st.columns(3) 
            for idx, food in enumerate(result.get("food_recommendations", [])):
                with f_cols[idx % 3]:
                    with st.container(border=True):
                        st.image(food['image'], use_container_width=True)
                        st.markdown(f"<h6 style='text-align: center;'>🍴 {food['name']}</h6>", unsafe_allow_html=True)

    st.toast("Travel plan finalized and formatted!", icon="✅")
