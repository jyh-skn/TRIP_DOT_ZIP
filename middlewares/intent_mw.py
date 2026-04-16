from __future__ import annotations

from typing import Any
from langchain.agents.middleware import AgentMiddleware

from schemas.agent_state import TravelAgentState
from services.intent_service import classify_intent_by_rule


class IntentRoutingMiddleware(AgentMiddleware[TravelAgentState]):
    state_schema = TravelAgentState

    def __init__(
        self,
        weather_tools: list | None = None,
        place_tools: list | None = None,
        schedule_tools: list | None = None,
        modify_tools: list | None = None,
        travel_tools: list | None = None,
        chat_tools: list | None = None,
        enable_tool_filtering: bool = True,
        debug: bool = True,
    ):
        self.weather_tools = weather_tools or []
        self.place_tools = place_tools or []
        self.schedule_tools = schedule_tools or []
        self.modify_tools = modify_tools or []
        self.travel_tools = travel_tools or []
        self.chat_tools = chat_tools or []
        self.enable_tool_filtering = enable_tool_filtering
        self.debug = debug

    def _extract_user_text(self, state: TravelAgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return ""

        last_message = messages[-1]

        # 보통 HumanMessage.content 사용
        if hasattr(last_message, "content"):
            content = last_message.content
            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return " ".join(text_parts).strip()

        return str(last_message).strip()

    def before_agent(
        self,
        state: TravelAgentState,
        runtime,
    ) -> dict[str, Any] | None:
        user_text = self._extract_user_text(state)
        result = classify_intent_by_rule(user_text)

        if self.debug:
            print("[IntentRoutingMiddleware] user_text =", user_text)
            print("[IntentRoutingMiddleware] result =", result)

        return {
            "intent": result["intent"],
            "confidence": result["confidence"],
            "route": result["route"],
            "intent_reason": result["reason"],
        }

    def before_model(
        self,
        state: TravelAgentState,
        runtime,
    ) -> dict[str, Any] | None:
        if not self.enable_tool_filtering:
            return None

        route = state.get("route", "chat")

        route_to_tools = {
            "weather": self.weather_tools,
            "place": self.place_tools,
            "schedule": self.schedule_tools,
            "modify": self.modify_tools,
            "travel": self.travel_tools,
            "chat": self.chat_tools,
        }

        selected_tools = route_to_tools.get(route, self.chat_tools)

        if self.debug:
            tool_names = [getattr(tool, "name", str(tool)) for tool in selected_tools]
            print("[IntentRoutingMiddleware] route =", route)
            print("[IntentRoutingMiddleware] selected_tools =", tool_names)

        return {
            "tools": selected_tools
        }

    def before_agent(self, state, runtime):
        print("🔥🔥 middleware 들어옴 🔥🔥")