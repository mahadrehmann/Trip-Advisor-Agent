'''
Section 1: Simple Agent 
Scenario: 
    Build a Packing List Advisor Agent AI agent that: 
    Checks weather forecasts for a destination 
    Recommends packing items based on weather conditions 
    Suggests indoor/outdoor activities depending on rain/sun 
    Agents perceive the input and then decide which tool to use, either 'weather tool' or 'packing advisor tool'. Based on the weather conditions in the specified location, it recommends a packing list, or the user can also only ask for weather in a specific location. 

    You should be implementing a class Agent with 5 functions: 
    perceive which accepts the input and parses it to be sent to think function 
    think which decides which tool to use 
    2 functions for tools: weather tool and packing advisor tool 
    goal which outputs the result

    Task: 
    Implement an Agent class with the following specifications: 
    Perceive Function: 
    Extracts location and intent from user input 
    Example input: "What should I pack for Paris?" → {"location": "paris", "intent": "packing"}
    Think Function:
    Decides which tools to use 
    If intent is "packing" → use both weather and packing tools 
    Else → use only weather tool 

    Tools: 
    weather_tool(location): Returns mock data like {"condition": "rainy", "temp": 14} 
    packing_advisor(weather_condition): Returns items like ["umbrella", "jacket"] 
    Goal Function: 
    Returns structured output 
    Example output for packing request: 
    { 
        "location": "paris", 
        "weather": {"condition": "rainy", "temp": 14}, 
        "packing_list": ["umbrella", "jacket"] 
    } 
'''

from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
from langchain.agents import AgentType, initialize_agent
import requests


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=300
)
weather_api_key = os.getenv("WEATHER_API_KEY")

def get_weather(place):
    # The API endpoint
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={place}&aqi=no"
    response = requests.get(url)

    # Print the response
    print(response.json())
    return response.json()



def get_advice(weather):
    return llm.invoke(f"Based on this weather info: {weather}, please suggest what items i should pack? Output Format: Item1, Item2, Item3...")


weather_tool = Tool(
    name="Weather Getting Tool",
    func=get_weather,
    description="Used to retrieve the weather of any place"
)

advice_tool = Tool(
    name="Packing Advisor Tool",
    func=get_advice,
    description="Used to get advice on what to pack based on weather"
)


tools = [weather_tool, advice_tool]

agent = initialize_agent(
    tools, 
    llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,    #Reason then act
    verbose = True
)


class TripAgent:
    def __init__(self):
        pass

        # REACT FRAMEWOKR
    def percieve(self):
        prompt = input("Hello! How can I help you today?")
        # print("giving this to think", prompt, "\n----------")
        self.think(prompt)

    def think(self, prompt):
        answer = llm.invoke(f"Given this prompt: {prompt}, Does the User Require Just Weather info or Do they also want packing advice too? Ouput Format: User wants info about the weather info of XYZ...")
       
        # print("giving this to goal", answer, "\n----------------")
        self.goal(answer)

    def goal(self, answer):
        syntax = '''Example output for packing request: 
                { 
                    "location": "paris", 
                    "weather": {"condition": "rainy", "temp": 14}, 
                    "packing_list": ["umbrella", "jacket"] 
                } 
                '''
        result = agent.invoke(f"If the user wants {answer}, tell me the output in this format: {syntax}" )

        print("Final Answer: \n", result)


ta = TripAgent()
ta.percieve()

'''
prompt: 
i need weather of peshawar

ouptut:

> Entering new AgentExecutor chain...
I need to find out the current weather information for Peshawar. 

Action: Weather Getting Tool  
Action Input: Peshawar  
{'location': {'name': 'Peshawar', 'region': 'North-West Frontier', 'country': 'Pakistan', 'lat': 34.0078, 'lon': 71.5733, 'tz_id': 'Asia/Karachi', 'localtime_epoch': 1751350213, 'localtime': '2025-07-01 11:10'}, 'current': {'last_updated_epoch': 1751349600, 'last_updated': '2025-07-01 11:00', 'temp_c': 33.3, 'temp_f': 91.9, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 41, 'wind_dir': 'NE', 'pressure_mb': 999.0, 'pressure_in': 29.5, 'precip_mm': 0.01, 'precip_in': 0.0, 'humidity': 56, 'cloud': 50, 'feelslike_c': 36.5, 'feelslike_f': 97.7, 'windchill_c': 36.1, 'windchill_f': 96.9, 'heatindex_c': 42.3, 'heatindex_f': 108.1, 'dewpoint_c': 22.7, 'dewpoint_f': 72.9, 'vis_km': 5.0, 'vis_miles': 3.0, 'uv': 9.0, 'gust_mph': 5.9, 'gust_kph': 9.4}}    

Observation: {'location': {'name': 'Peshawar', 'region': 'North-West Frontier', 'country': 'Pakistan', 'lat': 34.0078, 'lon': 71.5733, 'tz_id': 'Asia/Karachi', 'localtime_epoch': 1751350213, 'localtime': '2025-07-01 11:10'}, 'current': {'last_updated_epoch': 1751349600, 'last_updated': '2025-07-01 11:00', 'temp_c': 33.3, 'temp_f': 91.9, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 41, 'wind_dir': 'NE', 'pressure_mb': 999.0, 'pressure_in': 29.5, 'precip_mm': 0.01, 'precip_in': 0.0, 'humidity': 56, 'cloud': 50, 'feelslike_c': 36.5, 'feelslike_f': 97.7, 'windchill_c': 36.1, 'windchill_f': 96.9, 'heatindex_c': 42.3, 'heatindex_f': 108.1, 'dewpoint_c': 22.7, 'dewpoint_f': 72.9, 'vis_km': 5.0, 'vis_miles': 3.0, 'uv': 9.0, 'gust_mph': 5.9, 'gust_kph': 9.4}}
Thought:I have retrieved the current weather information for Peshawar. 

Final Answer: {
    "location": "Peshawar",
    "weather": {
        "condition": "Partly cloudy",
        "temp": 33.3
    }
}

'''


'''
prompt:
I have to got Naran, what should i pack?

ouput:

> Entering new AgentExecutor chain...
Thought: I need to first get the weather information for Naran. 
Action: Weather Getting Tool
Action Input: Naran{'location': {'name': 'Narang', 'region': 'Punjab', 'country': 'Pakistan', 'lat': 31.9103, 'lon': 74.5136, 'tz_id': 'Asia/Karachi', 'localtime_epoch': 1751349615, 'localtime': '2025-07-01 11:00'}, 'current': {'last_updated_epoch': 1751348700, 'last_updated': '2025-07-01 10:45', 'temp_c': 30.0, 'temp_f': 86.0, 'is_day': 1, 'condition': {'text': 'Overcast', 'icon': '//cdn.weatherapi.com/weather/64x64/day/122.png', 'code': 1009}, 'wind_mph': 10.1, 'wind_kph': 16.2, 'wind_degree': 116, 'wind_dir': 'ESE', 'pressure_mb': 1000.0, 'pressure_in': 29.53, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 79, 'cloud': 75, 'feelslike_c': 36.5, 'feelslike_f': 97.7, 'windchill_c': 29.1, 'windchill_f': 84.4, 'heatindex_c': 34.3, 'heatindex_f': 93.7, 'dewpoint_c': 24.6, 'dewpoint_f': 76.3, 'vis_km': 4.0, 'vis_miles': 2.0, 'uv': 4.5, 'gust_mph': 20.0, 'gust_kph': 32.2}}

Observation: {'location': {'name': 'Narang', 'region': 'Punjab', 'country': 'Pakistan', 'lat': 31.9103, 'lon': 74.5136, 'tz_id': 'Asia/Karachi', 'localtime_epoch': 1751349615, 'localtime': '2025-07-01 11:00'}, 'current': {'last_updated_epoch': 1751348700, 'last_updated': '2025-07-01 10:45', 'temp_c': 30.0, 'temp_f': 86.0, 'is_day': 1, 'condition': {'text': 'Overcast', 'icon': '//cdn.weatherapi.com/weather/64x64/day/122.png', 'code': 1009}, 'wind_mph': 10.1, 'wind_kph': 16.2, 'wind_degree': 116, 'wind_dir': 'ESE', 'pressure_mb': 1000.0, 'pressure_in': 29.53, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 79, 'cloud': 75, 'feelslike_c': 36.5, 'feelslike_f': 97.7, 'windchill_c': 29.1, 'windchill_f': 84.4, 'heatindex_c': 34.3, 'heatindex_f': 93.7, 'dewpoint_c': 24.6, 'dewpoint_f': 76.3, 'vis_km': 4.0, 'vis_miles': 2.0, 'uv': 4.5, 'gust_mph': 20.0, 'gust_kph': 32.2}}     
Thought:I have successfully retrieved the weather information for Naran. Now, I need to get packing advice based on the current weather conditions.

Action: Packing Advisor Tool  
Action Input: {'condition': 'Overcast', 'temp': 30.0}  
Observation: content='Based on the weather information provided (overcast conditions and a temperature of 30.0°C), here are some items you should consider packing:\n\nSunscreen, Light jacket, Water bottle, Sunglasses, Comfortable shoes, Hat, Umbrella, Snacks' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 48, 'total_tokens': 99, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'finish_reason': 'stop', 'logprobs': None} id='run--07e42c7a-b9da-41fe-9704-3ed3fce4e96d-0'
Thought:I now know the final answer.  

Final Answer: { 
    "location": "Naran", 
    "weather": {"condition": "Overcast", "temp": 30.0}, 
    "packing_list": ["Sunscreen", "Light jacket", "Water bottle", "Sunglasses", "Comfortable shoes", "Hat", "Umbrella", "Snacks"] 
}

'''