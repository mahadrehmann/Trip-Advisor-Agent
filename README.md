# Weather-Packing-Agent

An intelligent travel assistant agent that uses weather APIs and reasoning to recommend both the weather forecast and what to pack for your destination. Built using LangChain and tool-based agent reasoning (ReAct).

---

## ✈️ Use Case

You can ask:
- “What’s the weather like in Peshawar?”
- “I’m going to Naran, what should I pack?”

The agent:
- Detects your intent (weather only vs. weather + packing)
- Retrieves weather using a tool
- (If needed) Calls an LLM-powered advisor tool to suggest what to pack
- Returns a structured and helpful response

---

## 🧠 Agent Workflow

1. **Perceive**: Understand the location and intent  
2. **Think**: Decide which tools to use (Weather tool / Packing tool / Both)  
3. **Act**: Run tools in sequence via LangChain's ReAct agent  
4. **Goal**: Return a final structured JSON-like response  

---

## ⚙️ Tools Used

- 🌤️ **Weather Tool** (via `weatherapi.com`)
- 🧳 **Packing Advisor Tool** (uses GPT-4o-mini via OpenAI)

---

## 📥 Sample Inputs & Outputs

### 🔸 Input 1:
> **Prompt**: `I need the weather of Peshawar`

### ✅ Output:
```json
{
  "location": "Peshawar",
  "weather": {
    "condition": "Partly cloudy",
    "temp": 33.3
  }
}
```

### 🔸 Input 2:
> **Prompt**: I have to go to Naran, what should I pack?

✅ Output:
```json
{
  "location": "Naran",
  "weather": {
    "condition": "Overcast",
    "temp": 30.0
  },
  "packing_list": [
    "Sunscreen", "Light jacket", "Water bottle", "Sunglasses",
    "Comfortable shoes", "Hat", "Umbrella", "Snacks"
  ]
}
```

### 🔸 Input 3 (multi agent):
> **Prompt**: I want a beach vacation

✅ Output:
```json
{
  "destination_recommendation": {
    "location": "Bali",
    "trip_type": "beach"
  },
  "weather_info": {
    "condition": "Patchy rain nearby",
    "temp": 28.6
  },
  "packing_list": [
    "Umbrella",
    "Light rain jacket",
    "Waterproof shoes",
    "Comfortable clothing",
    "Hat",
    "Sunscreen"
  ]
}
```

## 🧪 Local Run

### 1. Create a .env file with your API keys:
```
OPENAI_API_KEY = your_openai_key (You can also use Gemini)
WEATHER_API_KEY = your_weatherapi_key
```
# 2. Install dependencies
```
pip install langchain openai requests python-dotenv
```

# 3. Run the agent
```
python agent.py
```