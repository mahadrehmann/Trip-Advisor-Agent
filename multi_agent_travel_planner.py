'''
    Section 2: Multi-Agent System  
    Build a Multi-Agent Travel Planning System that combines two specialized agents: 
    Destination Recommender Agent that suggests travel destinations based on user preferences (beach, adventure, cultural trips) 
    Weather Packing Agent (from Section 1) that generates packing lists for the recommended destination 
    The system coordinates between agents to provide complete travel planning assistance 

    Task: 
    The first agent perceives user preferences and recommends destinations using a 'destination recommender tool'. The second agent takes this recommendation, checks weather conditions, and provides a comprehensive packing list. The multi-agent system orchestrates the workflow between both agents to deliver a complete travel planning solution. 
    Final output should contain: 
    Combines outputs from both agents 
    Returns comprehensive travel planning response 
    Example output: 
    json 
    { 
        "destination_recommendation": { 
        "location": "Bali", 
        "trip_type": "beach" 
        }, 
        "weather_info": { 
        "condition": "sunny", 
        "temp": 28 
        }, 
        "packing_list": ["sunscreen", "swimsuit", "sunglasses", "light_clothing"] 
    } 
 
    Implement the destination recommender agent in the following way: 
    Destination Recommender Agent Class: 
    Perceive Function: 
    Extracts trip type from the user input 
    Example input: "I want a beach vacation" → {"trip_type": "beach”} 

    Think Function: 
    Decides to use destination recommender tool based on trip type 
    Tools: 
    destination_recommender_tool(trip_type): Returns mock destinations like {"destination": "Bali", "type": "beach"} 
    Goal Function: 
    Returns structured destination recommendation 
    Example output: {"recommended_destination": "Bali", "trip_type": "beach"} 

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
    max_tokens=200
)

# load_dotenv()
# google_api_key = os.getenv("GEMINI_API_KEY")

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     google_api_key=google_api_key,
#     temperature=0.5)


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

    def percieve(self):
        prompt = input("Hello! How can I help you today?")
        print("giving this to think", prompt, "\n----------")
        return self.think(prompt)

    def think(self, prompt):

        answer = f"User Requires Weather info and packing advice for {prompt}"
        # print("giving this to goal", answer, "\n----------------")
        return self.goal(answer)

    def goal(self, answer):
        syntax = '''Example output for packing request: 
                { 
                    "location": "paris", 
                    "weather": {"condition": "rainy", "temp": 14}, 
                    "packing_list": ["umbrella", "jacket"] 
                } 
                '''
        result = agent.invoke(f"If the user wants {answer}, tell me the output in this format: {syntax}" )

        return result





def get_recommendation(trip_type):
    syntax = ''' {"destination": "Bali", "type": "beach"}  '''
    return llm.invoke(f"I want a trip of this type: {trip_type}, please suggest a place I can visit. Output Format:{syntax}")


recommendation_tool = Tool(
    name="Place Recommendation Tool",
    func=get_recommendation,
    description="Used to get a place for vacation on the basis of trip type"
)

tools = [recommendation_tool]

place_agent = initialize_agent(
    tools, 
    llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,    #Reason then act
    verbose = True
)


class PlaceAgent:
    def __init__(self):
        pass

    def percieve(self):
        prompt = input("What kind of a Trip do you want? ")
        # print("giving this to think", prompt, "\n----------")
        return self.think(prompt)

    def think(self, prompt):
        syntax = ''' {"trip_type": "beach"} '''
        answer = llm.invoke(f"Given this prompt: {prompt}, What kind of a vacation does the user want? Ouput Format: {syntax} ")
       
        # print("giving this to goal", answer, "\n----------------")
        return self.goal(answer)

    def goal(self, answer):
        syntax = ''' 
                    { 
                        "destination_recommendation": { 
                            "location": "Bali", 
                            "trip_type": "beach" 
                        }
                    } 
                '''
        result = place_agent.invoke(f"If the user wants {answer}, tell me the output in this Example format: {syntax}" )

        print("Final Answer: \n", result)
        return result
    

pa = PlaceAgent()
result = pa.percieve()


ta = TripAgent()
result = ta.think(result)
print("Final Answer: \n", result)


'''
What kind of a Trip do you want? I want a beach vacation


> Entering new AgentExecutor chain...
I need to recommend a beach destination along with the weather information and a packing list for the user. 

Action: Place Recommendation Tool  
Action Input: {"trip_type": "beach"}  

Observation: content='{"destination": "Bali", "type": "beach"}' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 51, 'total_tokens': 65, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_62a23a81ef', 'finish_reason': 'stop', 'logprobs': None} id='run--c9246ddf-1e94-4d21-a938-4484fa70ff81-0'
Thought:I have a beach destination recommendation, which is Bali. Now, I need to check the weather there to provide the user with the weather information.

Action: Weather Getting Tool  
Action Input: "Bali"  
{'location': {'name': 'Bali', 'region': 'North Sumatra', 'country': 'Indonesia', 'lat': -0.1395, 'lon': 98.186, 'tz_id': 'Asia/Jakarta', 'localtime_epoch': 1751350694, 'localtime': '2025-07-01 13:18'}, 'current': {'last_updated_epoch': 1751350500, 'last_updated': '2025-07-01 13:15', 'temp_c': 28.6, 'temp_f': 83.5, 'is_day': 1, 'condition': {'text': 'Patchy rain nearby', 'icon': '//cdn.weatherapi.com/weather/64x64/day/176.png', 'code': 1063}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 303, 'wind_dir': 'WNW', 'pressure_mb': 1011.0, 'pressure_in': 29.84, 'precip_mm': 0.09, 'precip_in': 0.0, 'humidity': 70, 'cloud': 81, 'feelslike_c': 32.1, 'feelslike_f': 89.7, 'windchill_c': 28.6, 'windchill_f': 83.5, 'heatindex_c': 32.1, 'heatindex_f': 89.7, 'dewpoint_c': 22.6, 'dewpoint_f': 72.7, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 11.2, 'gust_mph': 5.5, 'gust_kph': 8.8}}    

Observation: {'location': {'name': 'Bali', 'region': 'North Sumatra', 'country': 'Indonesia', 'lat': -0.1395, 'lon': 98.186, 'tz_id': 'Asia/Jakarta', 'localtime_epoch': 1751350694, 'localtime': '2025-07-01 13:18'}, 'current': {'last_updated_epoch': 1751350500, 'last_updated': '2025-07-01 13:15', 'temp_c': 28.6, 'temp_f': 83.5, 'is_day': 1, 'condition': {'text': 'Patchy rain nearby', 'icon': '//cdn.weatherapi.com/weather/64x64/day/176.png', 'code': 1063}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 303, 'wind_dir': 'WNW', 'pressure_mb': 1011.0, 'pressure_in': 29.84, 'precip_mm': 0.09, 'precip_in': 0.0, 'humidity': 70, 'cloud': 81, 'feelslike_c': 32.1, 'feelslike_f': 89.7, 'windchill_c': 28.6, 'windchill_f': 83.5, 'heatindex_c': 32.1, 'heatindex_f': 89.7, 'dewpoint_c': 22.6, 'dewpoint_f': 72.7, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 11.2, 'gust_mph': 5.5, 'gust_kph': 8.8}}
Thought:I have the current weather information for Bali, which indicates a temperature of approximately 28.6°C with patchy rain nearby. Now, I need to suggest a packing list suitable for this weather.

Action: Packing Advisor Tool  
Action Input: {'temp': 28.6, 'condition': 'Patchy rain nearby'}  

Observation: content='Umbrella, Light rain jacket, Waterproof shoes, Comfortable clothing, Hat, Sunscreen' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 52, 'total_tokens': 69, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'finish_reason': 'stop', 'logprobs': None} id='run--6b4f3b50-acec-48d5-8a2a-bed991db4cf6-0'
Thought:I now have all the necessary information to compile the final response for the user.

Final Answer: { 
    "destination_recommendation": { 
        "location": "Bali", 
        "trip_type": "beach" 
    }, 
    "weather_info": { 
        "condition": "Patchy rain nearby", 
        "temp": 28.6 
    }, 
    "packing_list": ["Umbrella", "Light rain jacket", "Waterproof shoes", "Comfortable clothing", "Hat", "Sunscreen"] 
}
'''


# Second Proper Ouput

'''
> Entering new AgentExecutor chain...
I need to find a suitable hiking destination based on the user's request for a trip type of "hiking". 

Action: Place Recommendation Tool  
Action Input: {"trip_type": "hiking"}  

Observation: content='{"destination": "Banff National Park, Canada", "type": "hiking"}' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 51, 'total_tokens': 69, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'finish_reason': 'stop', 'logprobs': None} id='run--155ccf0c-4927-403e-baa0-efffac2bf879-0'
Thought:I now have the information I need to provide a hiking destination recommendation. 

Final Answer: { 
    "destination_recommendation": { 
        "location": "Banff National Park, Canada", 
        "trip_type": "hiking" 
    } 
}

> Finished chain.
Final Answer:
 {'input': 'If the user wants content=\'{"trip_type": "hiking"}\' additional_kwargs={} response_metadata={\'token_usage\': {\'completion_tokens\': 8, \'prompt_tokens\': 41, \'total_tokens\': 49, \'completion_tokens_details\': {\'accepted_prediction_tokens\': 0, \'audio_tokens\': 0, \'reasoning_tokens\': 0, \'rejected_prediction_tokens\': 0}, \'prompt_tokens_details\': {\'audio_tokens\': 0, \'cached_tokens\': 0}}, \'model_name\': \'gpt-4o-mini-2024-07-18\', \'system_fingerprint\': \'fp_34a54ae93c\', \'finish_reason\': \'stop\', \'logprobs\': None} id=\'run--588568fe-00a0-405f-bcbc-12cdec782d8b-0\', tell me the output in this Example format:  \n                    { \n                        "destination_recommendation": { \n                            "location": "Bali", \n                            "trip_type": "beach" \n                        }\n                   
 } \n                ', 'output': '{ \n    "destination_recommendation": { \n        "location": "Banff National Park, Canada", \n        "trip_type": "hiking" \n    } \n}'}   


> Entering new AgentExecutor chain...
I need to gather the weather information for Banff National Park, Canada, since the user is interested in hiking there. After obtaining the weather, I will then get packing advice based on that weather.

Action: Weather Getting Tool  
Action Input: "Banff National Park, Canada"  {'location': {'name': 'Banff', 'region': 'Alberta', 'country': 'Canada', 'lat': 51.1667, 'lon': -115.5667, 'tz_id': 'America/Edmonton', 'localtime_epoch': 1751353630, 'localtime': '2025-07-01 01:07'}, 'current': {'last_updated_epoch': 1751353200, 'last_updated': '2025-07-01 01:00', 'temp_c': 12.8, 'temp_f': 55.0, 'is_day': 0, 'condition': {'text': 'Clear', 'icon': '//cdn.weatherapi.com/weather/64x64/night/113.png', 'code': 1000}, 'wind_mph': 2.2, 'wind_kph': 3.6, 'wind_degree': 108, 'wind_dir': 'ESE', 'pressure_mb': 1016.0, 'pressure_in': 29.99, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 57, 'cloud': 0, 'feelslike_c': 13.4, 'feelslike_f': 56.0, 'windchill_c': 12.3, 'windchill_f': 54.2, 'heatindex_c': 12.2, 'heatindex_f': 53.9, 'dewpoint_c': 5.0, 'dewpoint_f': 41.0, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 0.0, 'gust_mph': 6.7, 'gust_kph': 10.8}}

Observation: {'location': {'name': 'Banff', 'region': 'Alberta', 'country': 'Canada', 'lat': 51.1667, 'lon': -115.5667, 'tz_id': 'America/Edmonton', 'localtime_epoch': 1751353630, 'localtime': '2025-07-01 01:07'}, 'current': {'last_updated_epoch': 1751353200, 'last_updated': '2025-07-01 01:00', 'temp_c': 12.8, 'temp_f': 55.0, 'is_day': 0, 'condition': {'text': 'Clear', 'icon': '//cdn.weatherapi.com/weather/64x64/night/113.png', 'code': 1000}, 'wind_mph': 2.2, 'wind_kph': 3.6, 'wind_degree': 108, 'wind_dir': 'ESE', 'pressure_mb': 1016.0, 'pressure_in': 29.99, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 57, 'cloud': 0, 'feelslike_c': 13.4, 'feelslike_f': 56.0, 'windchill_c': 12.3, 'windchill_f': 54.2, 'heatindex_c': 12.2, 'heatindex_f': 53.9, 'dewpoint_c': 5.0, 'dewpoint_f': 41.0, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 0.0, 'gust_mph': 6.7, 'gust_kph': 10.8}}      
Thought:I have obtained the current weather information for Banff National Park, Canada. The temperature is 12.8°C (55°F), and the weather condition is clear. Now, I will proceed to get packing advice based on this weather.

Action: Packing Advisor Tool  
Action Input: {'condition': 'Clear', 'temp': 12.8}  
Observation: content='Light jacket, T-shirt, Comfortable shoes, Sunglasses, Water bottle, Hat, Sunscreen.' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 47, 'total_tokens': 67, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'finish_reason': 'stop', 'logprobs': None} id='run--b9957ef7-1e6c-459b-b565-417e6a6cd506-0'
Thought:I now know the final answer. 

Final Answer: { 
    "location": "Banff National Park, Canada", 
    "weather": {"condition": "Clear", "temp": 12.8}, 
    "packing_list": ["Light jacket", "T-shirt", "Comfortable shoes", "Sunglasses", "Water bottle", "Hat", "Sunscreen"] 
}
'''