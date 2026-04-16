from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from middlewares.intent_mw import IntentRoutingMiddleware
from schemas.agent_state import TravelAgentState

# tools.py에서 가져오기
from llm.tools import (
    get_weather_tool,
    search_place_tool,
    make_schedule_tool,
    modify_schedule_tool,
    recommend_travel_tool,
)
from config import Settings

def build_trip_agent():
    settings = Settings()
    settings.validate()

    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key
    )

    intent_middleware = IntentRoutingMiddleware(
        weather_tools=weather_tools,
        place_tools=place_tools,
        schedule_tools=schedule_tools,
        modify_tools=modify_tools,
        travel_tools=travel_tools,
        chat_tools=chat_tools,
        enable_tool_filtering=True,
        debug=True,
    )

    agent = create_agent(
        model=model,
        tools=all_tools,
        middleware=[intent_middleware],
        state_schema=TravelAgentState,
    )

    return agent

# 모델
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# intent별 tool 그룹 (핵심)
weather_tools = [get_weather_tool]
place_tools = [search_place_tool]
schedule_tools = [make_schedule_tool]
modify_tools = [modify_schedule_tool]
travel_tools = [recommend_travel_tool]
chat_tools = []  # 일반 대화는 tool 없이 처리

# 전체 tool (agent에는 전부 넣어야 함)
all_tools = (
    weather_tools
    + place_tools
    + schedule_tools
    + modify_tools
    + travel_tools
    + chat_tools
)

# middleware
intent_middleware = IntentRoutingMiddleware(
    weather_tools=weather_tools,
    place_tools=place_tools,
    schedule_tools=schedule_tools,
    modify_tools=modify_tools,
    travel_tools=travel_tools,
    chat_tools=chat_tools,
    enable_tool_filtering=True,
    debug=True,
)

# agent 생성
agent = create_agent(
    model=model,
    tools=all_tools,
    middleware=[intent_middleware],
    state_schema=TravelAgentState,
)

result = agent.invoke(
    {
        "messages": [
            ("user", "부산 이번 주말 날씨 어때?")
        ]
    }
)

print(result)