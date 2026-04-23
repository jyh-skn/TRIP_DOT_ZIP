from typing import Annotated, Dict, List, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


IntentType = Literal[
    "general_chat",
    "travel_recommendation",
    "place_search",
    "schedule_generation",
    "weather_query",
    "modify_request",
]


class QualityCheck(TypedDict):
    is_passed: bool
    issues: List[str]
    target_node: str


# 기존 값이 있다면 유지하고, 새로운 값이 들어올 때만 업데이트하는 함수
def keep_and_update(existing: str | None, new: str | None) -> str | None:
    # 새로운 값이 있으면 그것을 쓰고, 없으면 기존 값을 유지합니다.
    if new is not None:
        return new
    return existing


class TravelAgentState(TypedDict, total=False):
    # 기본 대화 / 라우팅
    messages: Annotated[list, add_messages]
    intent: IntentType
    confidence: float
    route: str

    # 여행 조건
    destination: Annotated[str | None, keep_and_update]
    styles: List[str]
    constraints: List[str]
    travel_date: str
    relative_days: int
    raw_date_text: str
    start_time: str

    # 장소 / 일정 / 날씨
    mapped_places: List[Dict]
    selected_places: List[Dict]
    itinerary: List[Dict]
    weather_data: Dict

    # 흐름 제어
    missing_slots: List[str]
    need_weather: bool
    state_type_cd: str
    quality_check: QualityCheck

    # 지도 / 응답
    map_metadata: Dict
    final_response: str

    # 안전 / 요약
    blocked: bool
    blocked_reason: str
    conversation_summary: str
    conversation_summarized: bool
