from langchain.tools import tool
from langchain_tavily import TavilySearch
from langsmith import traceable
from dotenv import load_dotenv
import requests
import os
load_dotenv()

weatherapi=os.getenv("OPENWEATHER_API_KEY")
search_tool=TavilySearch(max_results=3)

@tool
def get_weather(city:str)->dict:
     """this function used to fetch weather data of a city"""
     url = "https://api.openweathermap.org/data/2.5/weather"

     params = {
          "q": city,
          "appid": weatherapi,
          "units": "metric"
     }

     response=requests.get(url,params=params)

     if response.status_code != 200:
          return f"no data found for city {city}"

     data=response.json()

     return {
          "temperature":data["main"]["temp"],
          "feels like":data["main"]["feels_like"],
          "description":data["weather"][0]["description"]
     }


