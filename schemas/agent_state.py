from typing import Literal
from typing_extensions import NotRequired
from langchain.agents.middleware import AgentState

IntentType = Literal[
    "general_chat",
    "travel_recommendation",
    "place_search",
    "schedule_generation",
    "weather_query",
    "modify_request",
]


class TravelAgentState(AgentState):
    intent: NotRequired[IntentType]
    confidence: NotRequired[float]
    route: NotRequired[str]
    intent_reason: NotRequired[str]