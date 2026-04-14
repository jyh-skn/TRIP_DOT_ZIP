from openai import OpenAI
from middlewares.pipeline import LLMRequest, LLMResponse
import re

# =========================
# 1. Bad word 리스트
# =========================
BAD_WORDS = [
    "씨발", "시발", "ㅅㅂ",
    "병신", "븅신", "ㅄ",
    "개새끼", "ㅈ같", "좆", "fuck"
]

# =========================
# 1-1. Global moderation threshold
# =========================
GLOBAL_BLOCK_THRESHOLD = 0.7


def contains_bad_word(text: str) -> bool:
    """텍스트에 욕설이 포함되어 있는지 확인한다.

    Args:
        text (str): 검사할 입력 문자열

    Returns:
        bool: 욕설이 포함되어 있으면 True, 아니면 False
    """
    lowered = text.lower()
    return any(word in lowered for word in BAD_WORDS)


# =========================
# 2. Moderation API 호출
# =========================
def check_moderation(client: OpenAI, text: str) -> dict:
    """OpenAI Moderation API를 호출하여 유해성 여부를 판단한다.

    Args:
        client (OpenAI): OpenAI 클라이언트 인스턴스
        text (str): 검사할 입력 문자열

    Returns:
        dict: moderation 결과
            {
                "flagged": bool,
                "categories": dict,
                "scores": dict
            }
    """
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    result = response.results[0]

    return {
        "flagged": result.flagged,
        "categories": dict(result.categories),
        "scores": dict(result.category_scores),
    }


def should_block_by_score(category_scores: dict) -> bool:
    """카테고리와 상관없이 score가 전역 threshold를 넘는지 판단한다.

    Args:
        category_scores (dict): moderation category_scores 결과

    Returns:
        bool: 차단해야 하면 True, 아니면 False
    """
    for category, score in category_scores.items():
        if score >= GLOBAL_BLOCK_THRESHOLD:
            print(
                f"🚫 score 차단: {category}={score:.4f} "
                f"(threshold={GLOBAL_BLOCK_THRESHOLD})"
            )
            return True
    return False


# =========================
# 3. 차단 여부 판단
# =========================
def should_block(client: OpenAI, text: str) -> bool:
    """텍스트를 차단해야 하는지 여부를 판단한다.

    욕설만 포함된 경우는 soft 필터로 처리하고,
    moderation score 중 하나라도 threshold를 넘으면 차단한다.

    Args:
        client (OpenAI): OpenAI 클라이언트
        text (str): 검사할 입력 문자열

    Returns:
        bool: 차단해야 하면 True, 아니면 False
    """
    # 욕설만 있는 경우 → moderation API 호출 생략
    if contains_bad_word(text):
        print("⚠️ bad word 감지: soft 필터만 적용")
        return False

    mod = check_moderation(client, text)

    print("🔍 moderation flagged:", mod["flagged"])
    print("🔍 moderation categories:", mod["categories"])
    print("🔍 moderation scores:", mod["scores"])

    # score 기반 차단
    if should_block_by_score(mod["scores"]):
        return True

    return False


# =========================
# 4. Middleware 팩토리
# =========================
def profanity_middleware(openai_client: OpenAI):
    """욕설 및 유해 입력을 필터링하는 미들웨어를 생성한다.

    Args:
        openai_client (OpenAI): OpenAI 클라이언트 객체

    Returns:
        Callable: (request, next_) -> LLMResponse 형태의 middleware 함수
    """

    def middleware(request: LLMRequest, next_) -> LLMResponse:
        """Pipeline에서 실행되는 실제 미들웨어 로직.

        Args:
            request (LLMRequest): LLM 요청 객체
            next_ (Callable): 다음 middleware 또는 base_handler

        Returns:
            LLMResponse: 다음 단계에서 반환된 응답
        """
        user_texts = [
            m.get("content", "")
            for m in request.messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        full_text = " ".join(user_texts)

        print("🔥 middleware 실행됨")
        print("입력:", full_text)

        # Hard 차단
        if should_block(openai_client, full_text):
            print("🚫 차단됨")
            raise ValueError("땃쥐가 상처받아 뒤돌았습니다.")

        # Soft 필터
        if contains_bad_word(full_text):
            print("⚠️ 욕설 감지됨")
            for msg in request.messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    msg["content"] = f"[주의: 과격한 표현 포함]\n{msg['content']}"
            request.metadata["profanity_detected"] = True

        return next_(request)

    return middleware


# =========================
# 1. PII 패턴 정의
# =========================
PII_PATTERNS = {
    "PHONE": re.compile(r"\b01[0-9]-?\d{3,4}-?\d{4}\b"),
    "EMAIL": re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
    "CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "RRN": re.compile(r"\b\d{6}-?[1-4]\d{6}\b"),  # 주민등록번호 패턴
    "PASSPORT": re.compile(r"\b[A-Z]{1,2}\d{7,8}\b"),
    "ACCOUNT": re.compile(r"\b\d{2,4}-\d{2,6}-\d{2,6}\b"),
}

ADDRESS_KEYWORDS = [
    "서울", "부산", "인천", "대구", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    "아파트", "빌라", "번지", "로", "길", "동", "호"
]

ADDRESS_PATTERNS = [
    re.compile(r"\b[가-힣]+시\s?[가-힣]+구\s?[가-힣0-9]+(로|길)\s?\d+\b"),
    re.compile(r"\b[가-힣]+구\s?[가-힣0-9]+(로|길)\s?\d+\b"),
    re.compile(r"\b[가-힣]+동\s?\d{1,3}(-\d{1,3})?번지\b"),
]

HIGH_RISK_TYPES = {"RRN", "CARD", "ACCOUNT"}
MEDIUM_RISK_TYPES = {"PHONE", "EMAIL", "PASSPORT", "ADDRESS"}


# =========================
# 2. PII 탐지
# =========================
def detect_pii(text: str) -> list[dict]:
    """텍스트에서 개인정보를 탐지한다.

    Args:
        text (str): 검사할 문자열

    Returns:
        list[dict]: 탐지 결과 리스트
            [
                {
                    "type": "PHONE",
                    "value": "010-1234-5678",
                    "start": 10,
                    "end": 23,
                    "risk": "medium"
                },
                ...
            ]
    """
    detected = []

    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            risk = "high" if pii_type in HIGH_RISK_TYPES else "medium"
            detected.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "risk": risk,
            })

    # 주소 키워드 기반 단순 탐지
    for keyword in ADDRESS_KEYWORDS:
        if keyword in text:
            detected.append({
                "type": "ADDRESS",
                "value": keyword,
                "start": text.find(keyword),
                "end": text.find(keyword) + len(keyword),
                "risk": "medium",
            })
            break

    return detected


# =========================
# 3. 차단 여부 판단
# =========================
def should_block_pii(detected_entities: list[dict]) -> bool:
    """탐지된 개인정보 중 hard block 대상이 있는지 확인한다.

    Args:
        detected_entities (list[dict]): detect_pii 결과

    Returns:
        bool: 차단 여부
    """
    return any(entity["type"] in HIGH_RISK_TYPES for entity in detected_entities)


# =========================
# 4. 마스킹
# =========================
def redact_pii(text: str, detected_entities: list[dict]) -> str:
    """탐지된 개인정보를 placeholder로 치환한다.

    Args:
        text (str): 원본 문자열
        detected_entities (list[dict]): detect_pii 결과

    Returns:
        str: 마스킹된 문자열
    """
    redacted = text
    for entity in sorted(detected_entities, key=lambda x: x["start"], reverse=True):
        placeholder = f"[{entity['type']}]"
        redacted = redacted[:entity["start"]] + placeholder + redacted[entity["end"]:]
    return redacted


# =========================
# 5. Sanitizer
# =========================
def sanitize_pii(text: str) -> dict:
    """텍스트에서 개인정보를 탐지하고 차단 여부/마스킹 결과를 반환한다.

    Args:
        text (str): 원본 문자열

    Returns:
        dict: 처리 결과
    """
    detected = detect_pii(text)
    blocked = should_block_pii(detected)
    sanitized_text = redact_pii(text, detected)

    return {
        "original_text": text,
        "sanitized_text": sanitized_text,
        "detected_entities": detected,
        "blocked": blocked,
    }


# =========================
# 6. Middleware 팩토리
# =========================
def pii_middleware():
    """개인정보 필터링 middleware를 생성한다.

    Returns:
        Callable: (request, next_) -> LLMResponse
    """

    def middleware(request: LLMRequest, next_) -> LLMResponse:
        if not hasattr(request, "metadata") or request.metadata is None:
            request.metadata = {}

        user_texts = [
            m.get("content", "")
            for m in request.messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        full_text = " ".join(user_texts)

        print("🔐 PII middleware 실행됨")
        print("입력:", full_text)

        pii_result = sanitize_pii(full_text)
        detected = pii_result["detected_entities"]

        if detected:
            print("🔎 탐지된 개인정보:", detected)

        # Hard block
        if pii_result["blocked"]:
            print("🚫 민감 개인정보 차단됨")
            raise ValueError("민감한 개인정보가 포함되어 있어 요청이 차단되었습니다.")

        # Soft redact
        if detected:
            for msg in request.messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    msg["content"] = redact_pii(msg["content"], detect_pii(msg["content"]))

            request.metadata["pii_detected"] = True
            request.metadata["pii_entities"] = detected
            request.metadata["sanitized"] = True
        else:
            request.metadata["pii_detected"] = False
            request.metadata["sanitized"] = False

        return next_(request)

    return middleware