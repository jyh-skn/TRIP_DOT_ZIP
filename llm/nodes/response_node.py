# nodes/response_nodes.py
from llm.graph.state import TravelAgentState
from llm.graph.contracts import StateKeys

def build_response_node(state: TravelAgentState) -> dict:
    """
    모든 분석 결과를 종합하여 사용자에게 줄 최종 답변을 생성합니다.
    """
    # 1. 데이터 가져오기
    intent = state.get(StateKeys.INTENT)
    weather = state.get(StateKeys.WEATHER_DATA)
    places = state.get(StateKeys.MAPPED_PLACES, [])
    itinerary = state.get(StateKeys.ITINERARY, [])
    destination = state.get(StateKeys.DESTINATION, "요청하신 지역")

    # 2. 답변 시나리오 구성
    response_text = ""

    # (A) 날씨 정보가 있을 때
    if weather:
        temp = weather.get("temp", "알 수 없음")
        desc = weather.get("description", "정보 없음")
        response_text += f"☀️ **{destination} 날씨 정보**: 현재 기온은 {temp}도이며, {desc} 상태입니다.\n\n"

    # (B) 일정 정보가 있을 때 (최우선순위)
    if itinerary:
        response_text += f"📅 **{destination} 추천 일정**을 짜봤어요!\n"
        for item in itinerary:
            # scheduler_service의 데이터 구조에 맞춰 출력
            response_text += f"- {item['arrival']} ~ {item['departure']}: {item['place_name']}\n"
        response_text += "\n이 일정대로 이동하시면 효율적이에요! 마음에 드시나요?"

    # (C) 장소 추천만 있을 때
    elif places:
        response_text += f"📍 **{destination} 추천 장소**들이에요:\n"
        for p in places[:3]: # 상위 3개만 노출
            response_text += f"- {p['name']} ({p.get('category', '명소')})\n"
        response_text += "\n이 장소들을 중심으로 일정을 짜드릴까요?"

    # (D) 일반 대화 혹은 데이터 부족 시
    else:
        response_text = "죄송해요, 요청하신 정보를 찾지 못했습니다. 다시 말씀해 주시겠어요?"

    # 3. 결과 반환 (final_response 업데이트)
    return {StateKeys.FINAL_RESPONSE: response_text}