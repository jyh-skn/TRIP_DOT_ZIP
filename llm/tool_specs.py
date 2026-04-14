TOOLS = [
    {
        "type": "function",
        "name": "get_weather_forecast",
        "description": "특정 지역과 날짜의 날씨 예보를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "예: 서울, 부산, 제주"},
                "date": {"type": "string", "description": "YYYY-MM-DD 형식"}
            },
            "required": ["region", "date"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_places",
        "description": "지역, 테마, 예산, 날씨 조건에 맞는 장소 목록을 검색한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string"},
                "theme": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "예: cafe, museum, indoor, photo, local_food"
                },
                "budget_krw": {"type": "integer"},
                "weather_condition": {"type": "string"},
                "count": {"type": "integer", "default": 10}
            },
            "required": ["region", "theme", "budget_krw", "weather_condition"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "create_schedule",
        "description": "사용자가 선택하거나 추천된 장소들을 바탕으로 최적의 동선과 체류 시간을 계산하여 하루치 타임라인 일정을 생성합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "places": {
                    "type": "array",
                    "description": "방문할 장소들의 리스트. 첫 번째 요소는 여행의 시작점(예: 숙소)으로 간주합니다.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "장소 이름"},
                            "lat": {"type": "number", "description": "위도"},
                            "lng": {"type": "number", "description": "경도"},
                            "types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "구글 Places API 기준 장소 유형 리스트 (예: ['museum', 'park'])"
                            }
                        },
                        "required": ["name", "lat", "lng", "types"],
                        "additionalProperties": False
                    }
                },
                "start_time_str": {
                    "type": "string",
                    "description": "일정 시작 시각 (HH:MM 형식). 기본값 '09:00'",
                    "default": "09:00"
                },
                "mode": {
                    "type": "string",
                    "description": "이동 수단 설정",
                    "enum": ["transit", "walking", "driving"],
                    "default": "transit"
                },
                "optimize_route": {
                    "type": "boolean",
                    "description": "True이면 최적 동선으로 재배치하고, False이면 입력된 장소 순서를 유지합니다.",
                    "default": True
                }
            },
            "required": ["places"],
            "additionalProperties": False
        },
        "strict": True
    }
]