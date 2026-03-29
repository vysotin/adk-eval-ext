"""Weather agent — single agent with two tools.

A simple ADK agent that can check current weather and multi-day forecasts
for any city. Demonstrates single-agent + multi-tool pattern.
"""

from google.adk.agents import Agent


# --- Tools ---


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: City name (e.g., "London", "New York", "Tokyo").

    Returns:
        Dictionary with temperature, condition, and humidity.
    """
    # Simulated weather data
    weather_data = {
        "london": {"city": "London", "temp_c": 15, "condition": "Cloudy", "humidity": 78},
        "new york": {"city": "New York", "temp_c": 22, "condition": "Sunny", "humidity": 55},
        "tokyo": {"city": "Tokyo", "temp_c": 28, "condition": "Humid", "humidity": 85},
        "paris": {"city": "Paris", "temp_c": 18, "condition": "Partly cloudy", "humidity": 65},
    }
    key = city.lower().strip()
    if key in weather_data:
        return weather_data[key]
    return {"city": city, "temp_c": 20, "condition": "Unknown", "humidity": 50}


def get_forecast(city: str, days: int = 3) -> dict:
    """Get a multi-day weather forecast for a city.

    Args:
        city: City name (e.g., "London", "New York").
        days: Number of forecast days (1-7, default 3).

    Returns:
        Dictionary with city name and list of daily forecasts.
    """
    days = min(max(days, 1), 7)
    # Simulated forecast
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly cloudy", "Windy", "Clear", "Stormy"]
    forecast = []
    for i in range(days):
        forecast.append({
            "day": i + 1,
            "temp_high_c": 20 + i * 2,
            "temp_low_c": 12 + i,
            "condition": conditions[i % len(conditions)],
        })
    return {"city": city, "days": days, "forecast": forecast}


# --- Agent ---


root_agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash",
    description="A weather assistant that provides current conditions and multi-day forecasts for cities worldwide.",
    instruction="""You are a helpful weather assistant. You can:
1. Check current weather conditions for any city using the get_weather tool
2. Get multi-day forecasts using the get_forecast tool

When a user asks about weather:
- If they want current conditions, use get_weather
- If they want a forecast, use get_forecast (default 3 days unless they specify)
- If they ask generally about weather, provide both current conditions and a short forecast
- Always present the information in a friendly, readable format
- Include temperature in Celsius
""",
    tools=[get_weather, get_forecast],
)
