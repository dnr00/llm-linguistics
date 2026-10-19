# llm-linguistics

한국어 형태론·화용론·형식 의미론 규칙을 결정론적으로 계산하고 온라인 코퍼스에서 도메인 지식을 학습하는 Python 라이브러리 및 MCP(Model Context Protocol) 서버입니다.

대형 언어 모델(LLM)은 BPE 토크나이저 구조 때문에 한글 음절·자모 단위 제약 검증에 취약하고, 한국어 어미 변화에 담긴 미시적 화계(높임법) 전환이나 복잡한 약관 문장의 논리적 모순을 확률적 추측에 의존합니다. `llm-linguistics`는 이러한 영역을 외부 LLM API 호출 없이 로컬 CPU(`kiwipiepy`, `z3-solver`, 유니코드 연산)로 정확하게 처리해 에이전트에 반환합니다.

---

## 주요 기능과 아키텍처

패키지는 싱글톤 `Kiwi` 형태소 엔진을 공유하는 4개 하위 모듈과 `FastMCP` 기반 통합 서버로 구성됩니다.

| 모듈 | 역할 | 핵심 동작 |
| --- | --- | --- |
| `morph` | 형태론·음절 규칙 검증 및 RAG 정규화 | 음절·공백 제외 글자 수, 받침 규칙(무받침·유받침·울림소리), 초성 시퀀스, 모음조화 검사 / `ㄹ` 받침 예외를 반영한 10종 조사 결합 / 불규칙 활용 용언 원형 복원 및 복합명사 분해 |
| `pragmatics` | 화용론·화계 전이 및 간접화행 분석 | 어말어미(`EF`)·선어말어미(`-시-`)·보조사(`요`) 기반 6단계 화계 판정 / 화자 내 화계 강등 및 화자 간 비대칭 탐지 / 완곡한 거절·독촉·유보 등 발화 내적 행위(Illocutionary Force) 추론 |
| `semantics` | 형식 의미론 및 Z3 모순 검증 | 문장 내 양화사(`모든`/`일부`)·배타 보조사(`만`/`뿐`/`외에는`)·이중 부정(`-지 않을 수 없다`)·수치 조건 추출 / 자연어 약관 문장을 정형 명제로 자동 변환 후 `Z3 Solver`로 양상 충돌과 반례(Witness) 증명 |
| `learner` | 온라인 코퍼스·피드백 적응 | 웹 URL·RSS 피드 및 텍스트 배치에서 미등록 복합명사(OOV)를 발굴해 `Kiwi`에 즉시 반영 / 베이지안(Beta-Binomial) 사후 확률로 담화 표지 가중치 갱신 및 오탐 억제 / 형태소 어간 기준 양상 패턴 학습 |

---

## 설치 및 테스트

Python 3.12 환경과 `uv` 사용을 권장합니다.

```bash
git clone https://github.com/dnr00/llm-linguistics.git
cd llm-linguistics
uv sync --python 3.12 --extra dev

# 전체 테스트 실행
PYTHONNOUSERSITE=1 uv run pytest
```

---

## MCP 서버 연동

표준 입출력(`stdio`) 기반 MCP 프로토콜을 따르므로 Claude, OpenAI Codex, Gemini, Cursor 등 모든 MCP 클라이언트에서 동일하게 작동합니다.

### Claude Code (`claude`)
```bash
claude mcp add llm-linguistics -- uv --directory /path/to/llm-linguistics run llm-linguistics-mcp
```

### OpenAI Codex CLI (`~/.codex/config.toml`)
```toml
[mcp_servers.llm-linguistics]
command = "uv"
args = ["--directory", "/path/to/llm-linguistics", "run", "llm-linguistics-mcp"]
```

### Gemini CLI / Antigravity / Cursor (`settings.json` 또는 `mcp_config.json`)
```json
{
  "mcpServers": {
    "llm-linguistics": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/llm-linguistics",
        "run",
        "llm-linguistics-mcp"
      ]
    }
  }
}
```

---

## 제공 MCP 도구 (12종)

### 1. 형태론 (`morph_*`)
* `morph_verify_constraints`: 한글 음절 수, 공백 제외 글자 수, 받침 제약(`none` | `all` | `only_sonorant`), 초성 시퀀스(`expected_choseong_sequence`), 어절별 모음조화 준수 여부를 검사하고 위반한 정확한 음절 인덱스를 반환합니다.
* `morph_attach_josa`: 단어 마지막 음절의 종성 유무와 `ㄹ` 받침 여부를 판별해 `은/는`, `이/가`, `을/를`, `과/와`, `으로/로`, `이에요/예요` 등 10종 조사를 결합합니다.
* `morph_normalize_for_rag`: `ㅂ/ㄷ` 등 불규칙 활용 용언(`추워서` → `춥다`, `걸었다` → `걷다`)을 기본형으로 복원하고 복합어를 하위 어근으로 쪼개어 Hybrid RAG 인덱싱용 쿼리를 생성합니다.

### 2. 화용론 (`pragmatics_*`)
* `pragmatics_analyze_discourse`: 대화 턴(`speaker`, `text`) 목록을 분석해 발화별 6단계 화계(`formal_high` 하십시오체 ~ `informal_low` 해체), 주체 높임(`-시-`), 화계 강등 지점, 간접화행(`indirect_refusal_or_deferral`, `indirect_request_or_urge`, `hedged_dissent_or_reservation`) 및 신뢰도 점수를 반환합니다.

### 3. 의미론 (`semantics_*`)
* `semantics_extract_logical_markers`: 문장별 전칭/존재 양화사, 배타 표현, 이중 부정 여부, 의무·금지·허용 양상(`MUST` | `MUST_NOT` | `MAY`), 수치 비교 구문(`7일 이내` → `<= 7.0`)을 추출합니다.
* `semantics_verify_rule_consistency`: 정형화된 규칙(`PolicyRule`) 목록을 `Z3 Solver`로 검증해 도달 불가능한 규칙(Dead rule)과 양상 충돌 쌍, 구체적인 반례 변수 할당(`witness_assignment`)을 반환합니다.
* `semantics_auto_verify_sentences`: 별도의 수식 변환 없이 한국어 약관·규정 원문 문장 리스트만 입력받아 행위 명사, 조건절 한정 명사구, 수치 범위를 자동 파싱한 뒤 Z3로 모순을 검증합니다.

### 4. 온라인 학습 (`learn_*`)
* `learn_from_online_url`: 외부 웹페이지(HTML)나 뉴스·기술 블로그 RSS/Atom 피드 URL에서 한국어 문장을 수집해 도메인 미등록어(OOV)를 자동 발굴하고 런타임 `Kiwi` 사전에 즉시 반영합니다.
* `learn_domain_corpus`: 텍스트 배열에서 품사 경계 점수(`pos_score`)와 형태소 분할 여부를 기준으로 신조어·복합명사를 추출해 사전에 등록합니다.
* `learn_domain_word`: 단일 고유명사·전문용어를 지정 품사(`NNP` | `NNG` | `MAG` | `IC`)로 즉시 등록합니다.
* `learn_pragmatic_feedback`: 특정 담화 표지에 대한 정탐(`is_positive=True`)·오탐(`is_positive=False`) 피드백을 누적해 라플라스 스무딩 사후 확률을 갱신합니다. 사후 확률이 `0.55` 이상이면 활성 표지로 채택되고, `0.35` 이하로 떨어지면 기본 사전 표지라도 해당 도메인에서 억제됩니다.
* `learn_semantic_pattern`: 도메인 특화 의무·금지·허용 및 배타 조건 구문을 등록합니다. 표면 문자열뿐 아니라 `Kiwi` 형태소 어간 시퀀스로도 대조하므로 모든 활용형(`득한`, `득하여`, `득해야` 등)을 인식합니다.

---

## Python 라이브러리 사용 예시

```python
from llm_linguistics import (
    DialogueTurnInput,
    MorphConstraintSpec,
    analyze_discourse,
    attach_josa,
    normalize_for_rag,
    verify_constraints,
)
from llm_linguistics.semantics import auto_verify_korean_sentences

# 1. 음절·자모 제약 검증 및 조사 결합
check = verify_constraints(
    "바나나 토마",
    MorphConstraintSpec(exact_syllable_count=5, batchim_rule="none", expected_choseong_sequence="ㅂㄴㄴㅌㅁ"),
)
assert check.is_valid is True
assert attach_josa("서울", "으로/로").combined == "서울로"
assert attach_josa("부산", "으로/로").combined == "부산으로"

# 2. 불규칙 용언 원형 복원 (RAG 인덱싱)
rag = normalize_for_rag("날씨가 추워서 아름다운 거리를 빠르게 걸었다.")
# rag.content_lemmas -> ['날씨', '춥다', '아름답다', '거리', '빠르다', '걷다']

# 3. 화용론 담화 분석 (간접 거절 및 완곡한 독촉 탐지)
discourse = analyze_discourse([
    DialogueTurnInput(speaker="Vendor", text="혹시 언제쯤 일정 공유 가능하실까요?"),
    DialogueTurnInput(speaker="Client", text="내부적으로 논의 후 추후에 검토해 보겠습니다."),
])
# discourse.utterances[0].illocutionary_force == "indirect_request_or_urge"
# discourse.utterances[1].illocutionary_force == "indirect_refusal_or_deferral"

# 4. 자연어 약관 문장 Z3 자동 모순 검증
report = auto_verify_korean_sentences([
    "결제 후 7일 이내인 모든 주문은 반드시 환불해야 한다.",
    "디지털 콘텐츠는 결제 후 30일 이내라도 환불이 불가하다.",
])
# report.consistency_report.is_consistent == False
# report.consistency_report.conflicts[0].witness_assignment -> {'cond_디지털_콘텐츠': True, 'num_일': 0.0}
```

---

## 온라인 학습 상태 저장소

학습된 사용자 사전, 담화 표지 가중치, 의미론 패턴은 기본적으로 `~/.llm_linguistics/learned_state.json`에 원자적(Atomic rename)으로 저장되며 서버 재시작 시 자동 로드됩니다. 저장 경로를 바꾸려면 `LLM_LINGUISTICS_STATE_PATH` 환경변수를 지정하면 됩니다.
