"""Travel multi-agent — coordinator with two specialist sub-agents.

A multi-agent ADK application where a coordinator delegates to:
- flight_agent: searches and books flights (has search_flights tool)
- hotel_agent: searches and books hotels (has search_hotels tool)

Demonstrates multi-agent orchestration with tool delegation.
"""

from google.adk.agents import Agent


# --- Flight tools ---


def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search for available flights between two cities.

    Args:
        origin: Departure city (e.g., "London").
        destination: Arrival city (e.g., "Paris").
        date: Travel date in YYYY-MM-DD format.

    Returns:
        Dictionary with list of available flights.
    """
    # Simulated flight data
    flights = [
        {
            "flight_id": f"FL-{origin[:2].upper()}{destination[:2].upper()}-101",
            "airline": "SkyAir",
            "origin": origin,
            "destination": destination,
            "date": date,
            "departure": "08:00",
            "arrival": "10:30",
            "price_usd": 250,
        },
        {
            "flight_id": f"FL-{origin[:2].upper()}{destination[:2].upper()}-202",
            "airline": "CloudJet",
            "origin": origin,
            "destination": destination,
            "date": date,
            "departure": "14:00",
            "arrival": "16:30",
            "price_usd": 180,
        },
    ]
    return {"origin": origin, "destination": destination, "date": date, "flights": flights}


# --- Hotel tools ---


def search_hotels(city: str, check_in: str, check_out: str) -> dict:
    """Search for available hotels in a city.

    Args:
        city: City name (e.g., "Paris").
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.

    Returns:
        Dictionary with list of available hotels.
    """
    # Simulated hotel data
    hotels = [
        {
            "hotel_id": f"HT-{city[:3].upper()}-01",
            "name": f"Grand {city} Hotel",
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "price_per_night_usd": 150,
            "rating": 4.5,
        },
        {
            "hotel_id": f"HT-{city[:3].upper()}-02",
            "name": f"{city} Budget Inn",
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "price_per_night_usd": 75,
            "rating": 3.8,
        },
    ]
    return {"city": city, "check_in": check_in, "check_out": check_out, "hotels": hotels}


# --- Sub-agents ---


flight_agent = Agent(
    name="flight_agent",
    model="gemini-2.0-flash",
    description="Specialist for searching and recommending flights. Delegate to this agent when the user needs flight information.",
    instruction="""You are a flight specialist. Use the search_flights tool to find flights.
When presenting results:
- Show flight ID, airline, times, and price
- Recommend the best option based on price and convenience
- Ask for confirmation before proceeding
""",
    tools=[search_flights],
)


hotel_agent = Agent(
    name="hotel_agent",
    model="gemini-2.0-flash",
    description="Specialist for searching and recommending hotels. Delegate to this agent when the user needs hotel information.",
    instruction="""You are a hotel specialist. Use the search_hotels tool to find hotels.
When presenting results:
- Show hotel name, price per night, and rating
- Recommend the best option based on value and rating
- Ask for confirmation before proceeding
""",
    tools=[search_hotels],
)


# --- Coordinator ---


root_agent = Agent(
    name="travel_coordinator",
    model="gemini-2.0-flash",
    description="A travel planning coordinator that delegates to flight and hotel specialists.",
    instruction="""You are a travel planning coordinator. You help users plan trips by coordinating
flight and hotel bookings.

Your specialists:
- flight_agent: searches for flights between cities
- hotel_agent: searches for hotels in a city

When a user wants to plan a trip:
1. Understand their travel requirements (destination, dates, preferences)
2. Delegate flight search to flight_agent
3. Delegate hotel search to hotel_agent
4. Summarize the combined travel plan with total estimated cost

If the user only needs flights or only needs hotels, delegate to the appropriate specialist.
Always be helpful and proactive in suggesting complete travel plans.
""",
    sub_agents=[flight_agent, hotel_agent],
)
