"""
    FileName: validate_node.py
    Location: llm/nodes/validate_node.py
    Role: 검증자 노드: 여행 계획에 대한 검증 제공
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from test_backup.langgraph_jyhong.state import TempTravelAgentState, QualityCheck # QualityCheck는 테스트용
from pydantic import BaseModel, Field
from typing import List
from llm.graph.state import TravelAgentState


# 검증용 LLM 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 검증 결과 구조체
# state.py 내용을 pydantic 모델로 변경하면서 그거 가지고 올 예정.
class QualityCheckResult(BaseModel):
    is_passed: bool = Field(default=False, description="통과 여부")
    issues: List[str] = Field(description="실패 사유 리스트")
    target_node: str = Field(description="분기처리를 위한 다음 노드")

# 검증 프롬프트(위치는 추후 llm/prompts.py로 이동 예정)
VALIDATION_PROMPT = """
    당신은 베테랑 여행 플래너이자 검증자입니다. 

    아래 사용자의 요구사항과 추천 장소 리스트를 비교하여 엄격하게 품질을 평가해줘.
    1. 사용자의 요구사항 '{styles}'과 추천 장소가 얼마나 일치하는지 평가
    2. 제약 조건: '{constraints}'이 일정에 잘 반영되었는지 평가
    3. 논리적 타당성: 추천장소 리스트의 장소들이'{raw_places}'가 실제로 여행 경로 상 실제로 방문이 가능한 곳인지.    # 이 부분을
    3. ★시간 타당성: 각 장소의 'arrival'과 'departure' 사이의 간격이 충분한가? (식사 1시간, 관람 1.5~2시간 등)    # 3번과            ★
    4. ★동선 효율성: 장소 간 이동 시간이 현실적인가? 너무 멀리 떨어진 곳을 무리하게 배치하지 않았는가?              # 4번으로 바꾸면 어떨까요

    검증 결과:                                                                                                # 여기서부터
    - 동선이나 시간 배분이 문제라면 target_node를 "scheduler_node"로 지정하세요.
    - 장소 자체가 취향에 안 맞으면 target_node를 "place_node"로 지정하세요.                                      # 여기까지도 추가    ★

    만약 하나라도 부적합하다면 is_passed = False로 하고, issues에는 구체적으로 어떤 점이 문제인지 작성해줘.
    target_node는 문제가 발견된 경우 돌아가야 할 노드 이름을 작성해줘. (예: "PlaceSearchNode", "MakeScheduleNode")
"""

# 생각해보니 validation은 사용자에게 보여주기 전에 존재해야하므로, schedule 부르기 전이겠는데?
# TODO: 검증이 필요한 시점과 검증 기준을 명확히 정의한 후, 프롬프트와 로직을 조정할 것.

def validate_travel_plan_node(state: TempTravelAgentState) -> dict:
    """ 여행 계획에 대한 검증을 수행하는 노드 함수. 
        Args: TempTravelAgentState 객체
        Returns: 검증 결과가 포함된 딕셔너리
    """
    # 검증 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": VALIDATION_PROMPT},
        {"role": "user", "content": f"사용자 요구사항: {state.styles}, 제약조건: {state.constraints}, 목적지: {state.destination}, 추천 장소 리스트: {state.raw_places}"}
    ])

    # 구조화된 출력 도구 생성
    structured_output = llm.with_structured_output(QualityCheckResult)

    try :   # 예외처리를 여기서? 또는 밖에서?
        # Chain 실행
        chain = prompt | structured_output
        result = chain.invoke(state.model_dump())

        return {"quality_check": result.model_dump(), "state_type_cd": "04"}
    except Exception as e:
        # 예외 처리: 검증 실패로 간주, 재호출 유도
        # TODO: 이것도 좀 더 고민거리.
        print(f"Error in Validator: {e}")
        return {
            "quality_check": {
                "is_passed": False, 
                "issues": ["시스템 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요."],
                "target_node": "search_places"
            },
            "state_type_cd": "02" # 다시 검색 단계로 보내거나 에러 페이지로 유도
        }

# 스케쥴 부른 이후에 검증하려면 itinerary를 불러오는게 어떨까 싶습니다. 아래 수정된 함수 확인 부탁드립니다.
def validate_travel_plan_node(state: dict) -> dict: # 기존 TempTravelAgentState 대신 dict/TravelAgentState 호환
    try:
        # [수정] 기존 raw_places 대신 실제 시간표인 itinerary를 우선 참조하도록 변경
        itinerary = state.get("itinerary", [])
        styles = state.get("styles", [])
        constraints = state.get("constraints", [])

        # LLM에게 전달할 텍스트 구성
        prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT)
        
        # 모델 연결 (기존 구조 유지)
        structured_output = llm.with_structured_output(QualityCheckResult)
        chain = prompt | structured_output
        
        # [수정] builder.py의 state 구조에 맞춰 데이터 전달
        result = chain.invoke({
            "itinerary": itinerary,
            "styles": styles,
            "constraints": constraints
        })

        return {"quality_check": result.model_dump()}
        
    except Exception as e:
        print(f"Error in Validator: {e}")
        return {
            "quality_check": {
                "is_passed": False, 
                "issues": ["검증 과정에서 오류가 발생했습니다."],
                "target_node": "place_node"
            }
        }

# 검증 후의 분기 로직
def route_after_validation(state: TravelAgentState):
    """ 검증 노드(validate_node)의 결과에 따라 다음 진행 노드를 결정하는 라우터 함수. 
        Args: TravelAgentState 객체 (품질 검증 결과인 quality_check 포함)
        Returns: 다음으로 이동할 노드 이름 (place_node, scheduler_node, 또는 response_node)
    """
    quality_check = state.get("quality_check")
    
    # 검증 통과 못 했을 때
    if quality_check and not quality_check.get("is_passed", True):
        target = quality_check.get("target_node")
        # validate_node.py에서 정의한 target_node로 유연하게 보냄
        # (예: 장소가 별로면 "place_node", 동선이 꼬였으면 "scheduler_node")
        return target if target in ["place_node", "scheduler_node"] else "place_node"
    
    # 통과하면 최종 답변 노드로
    return "response_node"
    
# 테스트 케이스 실행
if __name__ == "__main__":
    # 실제 사용자가 입력한 것 같은 State 객체 생성
    mock_state = TempTravelAgentState(
        destination="부산",
        styles=["액티비티", "바다"],
        constraints=["예산은 인당 5만원 이하", "도보 이동 위주"],
        raw_places=[
            "1. 해운대 요트 투어 (인당 7만원)", 
            "2. 광안리 서핑 체험 (인당 6만원)"
        ]
    )

    print("--- [테스트 시작: State 객체 전달] ---")
    update_data = validate_travel_plan_node(mock_state)

    # 4. 결과 확인 (LangGraph가 업데이트를 수행하는 방식 시뮬레이션)
    print(f"\n[LLM 검증 결과]")
    print(f"통과 여부: {update_data['quality_check']['is_passed']}")
    print(f"발견된 문제: {update_data['quality_check']['issues']}")
    print(f"다음 상태 코드: {update_data['state_type_cd']}")

    # 실제 State 객체에 반영해보기
    mock_state.quality_check = QualityCheck(**update_data['quality_check'])
    mock_state.state_type_cd = update_data['state_type_cd']
    
    print(f"\n[최종 업데이트된 State 객체 상태]")
    print(mock_state)
