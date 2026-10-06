# 더미 AI 의도 모듈 / 감정 모듈 설명 및 의도 모듈 수정 가이드

이 문서는 더미 AI(`DummyAgent`)가 **어떻게 행동을 고르는지**를 설명하고, 그중에서도 **의도 모듈(INTENT module, 역할에 따른 행동 판단)** 을 어떻게 수정하는지를 구체적으로 안내합니다. 감정(emotion) 쪽은 요점만 정리하고, JSON 작성 상세는 기존 문서 [`dummy_ai_emotion_guide.md`](./dummy_ai_emotion_guide.md)로 연결합니다.

필드 이름, 클래스 이름, JSON/YAML 키는 코드가 인식하는 리터럴이므로 영어 그대로 씁니다. 설명은 한국어로 작성합니다.

---

## 1. 개요: 세 모듈이 협력한다

`DummyAgent`는 **비공개(role) 정보를 절대 보지 않고**, 공개 관측값(public observation)만으로 매 턴 행동 하나를 결정합니다. 결정은 세 개의 모듈이 역할을 나눠 맡습니다.

| 모듈 | 파일 | 하는 일 |
| --- | --- | --- |
| 역할 추정 (role_belief) | `bang_rl/agents/dummy_agent/role_belief.py` | 공개 정보만으로 각 좌석의 역할 분포(`(5, 5)` 믿음 행렬)를 추정 |
| 의도 모듈 (INTENT) | `bang_rl/agents/dummy_agent/intent_policy.py` | 자기 역할 + 공개 관측 + 역할 믿음으로 **합법 행동마다 점수를 매기고 하나를 고름** |
| 감정 모듈 (EMOTION) | `bang_rl/agents/dummy_agent/emotion_module.py` | 공개 사건에 따라 감정/관계 상태를 갱신하고, 행동 선택에 더해질 보정값을 제공 |

이들을 하나로 묶는 것이 `bang_rl/agents/dummy_agent/dummy_agent.py`의 `DummyAgent`입니다. 흐름은 다음과 같습니다.

1. `RoleBeliefEstimator`가 공개 관측으로 역할 믿음 행렬을 갱신한다.
2. `IntentModule`이 그 믿음과 공개 관측으로 **의도 점수(intent scores)** 길이 6 벡터를 만든다.
3. `EmotionModule`(그리고 관계 벡터)이 **감정 기여분(emotion contribution)** 길이 6 벡터를 만든다.
4. 둘을 더한 결합 점수에 **공개 행동 마스크(action mask)** 를 적용해 `ActionSelector`가 최종 행동을 고른다.

**행동 공간 계약(action space contract, GUI와 공유):** 길이 6. 인덱스 `0` = PASS, 인덱스 `1..5` = `player_0..player_4`를 쏘는 BANG. 즉 좌석 `s`를 쏘는 행동 인덱스는 `s + 1`입니다.

---

## 2. 의도 모듈 (INTENT): 역할에 따른 행동 판단

### 2.1 무엇을 하는가

의도 모듈은 "역할에 따른 행동 판단"을 담당합니다. 자기 역할(role_belief에서 추정한 self role)과 공개 관측(각 좌석의 체력 `healths`, 생존 `alive`, 보안관 좌석 `sheriff_idx` 등), 그리고 역할 믿음 행렬을 입력으로 받아, **합법 행동마다 점수를 매긴 뒤 가장 높은 점수의 행동을 고릅니다.** 정책은 결정적(deterministic)이며, 동점이면 가장 낮은 행동 인덱스를 택합니다.

점수화 논리는 자기 역할에 따라 달라집니다(모두 공개 믿음 행렬 기반).

- **보안관 팀(sheriff, deputy):** 각 대상이 무법자/배신자일 확률을 중심으로 점수를 올린다(`sheriff_team_attack_weight`, `outlaw_team_attack_weight`, `traitor_attack_weight` 등).
- **무법자(outlaw_a, outlaw_b):** 보안관 승리 목표를 노려 보안관/보안관보 확률에 가중한다(`sheriff_role_bonus`, `deputy_role_bonus`, `outlaw_team_attack_weight`).
- **배신자(traitor):** 공개 추정상 더 강한 진영을 공격하며, 1:1 결투 상황이면 보안관 직접 공격 가중을 바꾼다(`sheriff_direct_duel_weight` vs `sheriff_direct_non_duel_weight`).

마지막으로 역할과 무관하게 **좌석별 공통 보정** 두 가지가 모든 대상 점수에 더해집니다.

- `role_target_weights`: 그 좌석의 **가장 유력한 역할 이름**에 설정된 가중치를 더한다.
- `target_health_weight`: 그 좌석의 공개 체력에 비례한 값을 더한다.

### 2.2 관련 파일/클래스

| 심볼 | 역할 |
| --- | --- |
| `intent_policy.py` | 의도 모듈 전체가 들어 있는 파일 (env에 의존하지 않음) |
| `IntentPolicyConfig` | 행동 점수화에 쓰는 **동결 데이터클래스(frozen dataclass)** 형태의 설정. 아래 2.3 표의 노브들을 담는다 |
| `IntentModule` | 설정과 관측/믿음을 받아 `IntentDecision`(점수 벡터)을 만드는 클래스 |
| `IntentCategory` | 직렬화 가능한 의도 라벨 enum: `pass`, `attack_sheriff_team`, `attack_outlaw_team`, `attack_enemy`, `attack_strong_faction`, `attack_sheriff_duel` |
| `ActionSelector` / `ActionSelection` | 의도 점수 + 감정 기여분 + 마스크를 결합해 행동을 고르는 선택기. deterministic은 최고 점수(동점 시 최저 인덱스), stochastic은 마스크 적용 softmax 샘플링 |

### 2.3 `IntentPolicyConfig` 노브 전체 표

아래는 `intent_policy.py`의 `IntentPolicyConfig`에 실제로 존재하는 모든 노브입니다(없는 노브를 상상하지 마세요). 모든 가중치 필드는 유한(finite) 숫자여야 하며, `__post_init__`가 유한성과 `tie_policy` 값을 검증합니다. 숫자가 아니거나 무한대/NaN이면 생성 시 `ValueError`가 납니다.

| 노브 | 타입 | 기본값 | 의미와 행동 효과 |
| --- | --- | --- | --- |
| `schema_version` | 문자열 | `"1.0"` | 정책 스키마 버전. 공백 불가. 저장/불러오기 호환 관리용. |
| `tie_policy` | `"sheriff"` 또는 `"outlaw"` | `"sheriff"` | 진영 강세(faction strength)가 완전히 동률일 때 어느 진영을 "강한 진영"으로 볼지. 배신자(traitor)의 공격 대상 선택에 영향. 그 외 값이면 `ValueError`. |
| `pass_score` | 실수 | `0.0` | PASS(행동 인덱스 0)의 기본 점수. 다른 공격 점수가 낮을 때 패스를 상대적으로 더/덜 매력적으로 만든다. |
| `sheriff_team_attack_weight` | 실수 | `1.0` | 보안관 팀일 때, 대상이 적(무법자+배신자)일 확률에 곱해지는 공격 가중. 클수록 적 추정 좌석을 더 공격. |
| `outlaw_team_attack_weight` | 실수 | `1.0` | 무법자일 때 보안관/보안관보 점수에 곱해지는 가중. 보안관 팀일 때는 대상이 무법자 좌석일 확률에 더해지는 가중으로도 쓰인다. |
| `traitor_attack_weight` | 실수 | `1.0` | 배신자일 때 "현재 강한 진영"에 속할 확률에 곱해지는 가중. 보안관 팀일 때는 대상이 배신자일 확률에 더해지는 가중으로도 쓰인다. |
| `sheriff_direct_non_duel_weight` | 실수 | `0.25` | 배신자가 **결투가 아닌** 상황에서 보안관(역할 0) 확률에 더하는 직접 공격 가중. |
| `sheriff_direct_duel_weight` | 실수 | `2.0` | 배신자가 **1:1 결투** 상황에서 보안관 확률에 더하는 직접 공격 가중(보통 non-duel보다 크게). |
| `sheriff_role_bonus` | 실수 | `1.0` | 대상이 보안관(역할 0)일 확률에 더해지는 보너스. 보안관 팀/무법자 점수화에서 사용. |
| `deputy_role_bonus` | 실수 | `0.25` | 대상이 보안관보(역할 1)일 확률에 더해지는 보너스. |
| `target_health_weight` | 실수 | `0.0` | 대상의 공개 체력에 곱해 모든 대상 점수에 더하는 가중. 양수면 체력 높은 좌석을, 음수면 체력 낮은 좌석을 선호. 기본 `0.0`이면 체력은 점수에 영향 없음. |
| `relation_target_weight` | 실수 | `0.0` | 감정 모듈의 **상대별 관계 벡터**를 행동 점수에 반영하는 가중(아래 2.4 절). 기본 `0.0`이면 관계 항이 전혀 더해지지 않아 기존 동작과 동일. 이 항은 의도 점수가 아니라 `DummyAgent._emotion_contribution`에서 더해진다. |
| `role_target_weights` | `{문자열: 유한숫자}` | `{}`(빈 매핑) | 좌석의 **가장 유력한 역할 이름**(`sheriff`, `deputy`, `outlaw_a`, `outlaw_b`, `traitor`)에 설정한 가중치를 그 좌석의 공격 점수에 더한다. 특정 역할을 선호/기피하도록 조절. 비워 두면 영향 없음. |

> 참고: 조회용 역할 이름은 `IntentModule._role_name`이 쓰는 `("sheriff", "deputy", "outlaw_a", "outlaw_b", "traitor")` 순서(역할 인덱스 0..4)와 같습니다. `role_target_weights`의 키는 반드시 이 다섯 이름 중 하나여야 효과가 있습니다(없는 이름은 무시됨).

### 2.4 `role_target_weights`와 `relation_target_weight` 자세히

이 두 노브는 "어느 좌석을 쏠지"를 역할/관계 관점에서 직접 조절하므로 특히 중요합니다.

- **`role_target_weights`(역할 기준):** 각 생존 좌석의 역할 믿음 중 **가장 유력한 역할 이름**을 뽑아, 그 이름에 설정된 가중치를 그 좌석의 공격 점수에 **더합니다.** 예를 들어 `{"outlaw_a": 1.0}`이면, 가장 유력 역할이 `outlaw_a`로 추정되는 좌석의 공격 점수가 `+1.0`만큼 올라 그 좌석을 더 자주 쏘게 됩니다. 음수를 주면 그 역할로 보이는 좌석을 **기피**합니다.
- **`relation_target_weight`(관계 기준):** 감정 모듈이 쌓은 **좌석별 관계 벡터**(부호 있는 값, 음수 = 원한 / 양수 = 신뢰)를 행동 점수에 반영합니다. 좌석 `s`를 쏘는 행동(인덱스 `s + 1`)에 더해지는 항은 `-relation_target_weight * relation[s]`입니다. 앞의 마이너스 때문에, 가중치가 양수일 때 **원한(음수 관계)** 은 그 좌석을 쏘는 점수를 **올리고**, **신뢰(양수 관계)** 는 낮춥니다. 즉 "원한을 품은 상대를 쏘도록" 유도합니다. 관계 벡터를 어떻게 쌓는지는 감정 가이드 3.4/3.7 절을 참고하세요.

두 신호는 서로를 대체하지 않고 최종 결합 점수 단계에서 **가산적으로** 누적됩니다.

---

## 3. 감정 모듈 (EMOTION): 요약과 상세 가이드 연결

감정 모듈(`emotion_module.py`, `EmotionModule`)은 공개 사건만으로 **설정 가능한 N개 감정**과 **좌석별 관계 벡터**를 갱신하는 결정적 모듈입니다. 핵심만 요약하면 다음과 같습니다.

- **설정 가능한 감정:** 이름/초기값/변화율/회복율과 네 개의 갱신 규칙 맵(`event_updates`, `action_updates`, `reward_updates`, `terminal_updates`) 및 행동 선택 참고용 `effects` 맵을 가진다. 모든 감정 값은 항상 `[0, 1]`로 잘린다.
- **관계로 한정된(relationship-qualified) 키:** `bang_at_me`, `bang_by_me`, `player_eliminated_at_other` 등 "누가 누구에게"를 공개 `actor_idx`/`target_idx`로 구분하는 이벤트 키를 인식한다.
- **상대별 관계 벡터:** 감정 값과 **별개로** 저장되는 부호 있는 `[-1, 1]` 길이 5 벡터. `relation_updates`로 좌석별 원한/신뢰를 누적한다. 2.4 절의 `relation_target_weight`가 이 벡터를 행동 점수에 연결한다.
- **HP 버킷(bucket) 키:** `hp_low_threshold`/`hp_lethal_threshold`를 기준으로 공개 `hp_after`를 비교해 `self_low_hp`, `target_lethal` 같은 키를 발동한다.

> JSON/YAML로 감정을 **직접 작성/편집**하는 전체 규칙(필드 표, 키 조회 순서, 완성 예시)은 **[`dummy_ai_emotion_guide.md`](./dummy_ai_emotion_guide.md)** 를 참고하세요. 이 문서는 그 내용을 중복하지 않고 요약만 둡니다.

---

## 4. 역할 추정 (role_belief)

`role_belief.py`의 `RoleBeliefEstimator`는 공개 정보만으로 각 좌석의 역할 분포를 추정하는 **결정적** 추정기입니다. 숨겨진 환경 역할 배열은 절대 받지 않습니다.

- 확정된 역할(예: 공개된 `sheriff_idx`, 공개 역할 공개 이벤트의 `revealed_role`)은 one-hot으로 고정하고, 아직 모르는 좌석에는 남은 역할 수를 균등 분배합니다.
- 결과는 `beliefs` 속성으로 `(5, 5)` 믿음 행렬(각 행 합 = 1)을 반환합니다.
- 이 행렬이 그대로 `IntentModule.decision(observation, beliefs)`의 입력이 됩니다. 즉 역할 추정이 바뀌면 의도 점수화의 "대상이 어떤 역할일 확률"이 바뀌어, 2.1 절의 모든 역할별 점수화와 `role_target_weights`의 "가장 유력한 역할" 판정이 함께 바뀝니다.

---

## 5. 모듈 연결 (dummy_agent)

`dummy_agent.py`의 `DummyAgent.act(observation, action_mask, ...)`가 세 모듈을 묶습니다. 순서는 다음과 같습니다.

1. `self.role_estimator.update(observation)` -- 공개 관측으로 역할 믿음을 갱신.
2. `self._prepare_emotion_values(...)` -- 공개 이벤트/스텝/보드 체력으로 감정·관계 상태를 갱신.
3. `intent = self.intent_module.decision(observation, self.role_beliefs)` -- 의도 점수(길이 6) 생성.
4. `ActionSelector.select_with_info(intent.scores, mask, emotion_contribution=self._emotion_contribution(), ...)` -- 결합 후 선택.

**`_emotion_contribution`의 구성.** `DummyAgent._emotion_contribution()`는 길이 6 벡터를 만들어 반환합니다.

- 먼저 각 행동에 대해 감정 모듈의 `effects_for_action(action)`을 조회해, 현재 감정 값과 효과(effect)의 곱을 합산합니다(감정 `effects` 기여분).
- 그다음 `relation_target_weight`가 `0.0`이 아니면, 모듈 수준 헬퍼 `_relation_target_contribution(relation, weight, player_idx)`로 **좌석별 관계 항**을 더합니다. 이 헬퍼는 행동 공간(인덱스 0 = pass, 1..5 = 좌석 `s = action - 1`)을 알고 있어, 각 BANG 행동에 `-weight * relation[s]`를 넣습니다. `weight == 0.0`이면 모든 항이 정확히 `0.0`이라 기존 동작과 완전히 동일합니다.

**마스크의 권위(authoritative).** 감정/관계 항은 어디까지나 **점수 보정**일 뿐이며, 어떤 행동이 합법인지는 전적으로 공통 **행동 마스크(action mask)** 가 정합니다. `ActionSelector`는 결합 점수에 마스크를 적용해 불법 행동을 완전히 배제한 뒤 선택하므로, 가중치를 아무리 크게 줘도 마스크로 금지된 행동은 선택되지 않습니다. 또한 관계 항은 관계 벡터를 **읽기만** 하고(감정/관계 상태를 바꾸지 않음) 결정성과 직렬화 스키마를 유지합니다.

**설정은 어디서 오는가.** `DummyAgent`를 만드는 애플리케이션 계층(`bang_rl/app/services.py`의 `_dummy` 헬퍼)은 `policy_config=config.preset.intent_policy`를 넘깁니다. 즉 런타임 의도 정책은 `SystemConfig.preset.intent_policy`(= `config.example.yaml`의 `preset.intent_policy` 키)에서 비롯됩니다.

---

## 6. 의도 모듈 수정 방법 (HOW TO MODIFY THE INTENT MODULE)

의도 모듈을 바꾸는 길은 두 가지입니다.

1. **설정 노브만 바꾸기(코드 수정 불필요):** 2.3 절 표에 **이미 존재하는** 노브 값을 조정. 대부분의 "AI 성향 조절"은 여기서 끝납니다.
2. **코드 수정(새 점수화 항 추가):** 표에 없는 **새 노브/새 점수화 규칙**이 필요할 때. `intent_policy.py`를 직접 편집합니다.

설정 노브의 출처는 항상 `SystemConfig.preset.intent_policy`입니다. YAML에서는 `config.example.yaml`에 다음과 같이 `preset.intent_policy` 블록을 추가/수정합니다. (예시 파일에는 기본 `preset`이 생략되어 기본값이 쓰입니다. 노브를 바꾸려면 아래처럼 명시하세요.)

### 예시 A: 특정 역할을 우선적으로 쏘게 하기 (`role_target_weights`)

"무법자로 보이는 좌석을 더 자주 쏘고, 보안관으로 보이는 좌석은 피하게" 하고 싶다고 합시다. 코드 수정 없이 `role_target_weights`만 조정하면 됩니다.

`config.example.yaml`:

```yaml
preset:
  preset_id: default
  version: "1.0"
  intent_policy:
    role_target_weights:
      outlaw_a: 1.5
      outlaw_b: 1.5
      sheriff: -1.0
```

의미: 가장 유력 역할이 `outlaw_a`/`outlaw_b`로 추정되는 좌석은 공격 점수 `+1.5`, `sheriff`로 추정되는 좌석은 `-1.0`. 키는 반드시 `sheriff`, `deputy`, `outlaw_a`, `outlaw_b`, `traitor` 중 하나여야 합니다(2.3 절). 값은 유한 숫자여야 합니다.

### 예시 B: 원한을 품은 좌석을 더 쏘게 하기 (`relation_target_weight`)

"나를 쏜 좌석에 원한을 쌓고, 그 좌석을 더 겨냥"하게 하려면 (1) 감정 설정에서 관계 벡터를 쌓고 (2) 의도 정책의 `relation_target_weight`를 `0`보다 크게 둡니다.

`config.example.yaml`:

```yaml
preset:
  preset_id: default
  version: "1.0"
  intent_policy:
    relation_target_weight: 1.5
  emotion:
    schema_version: "1.1"
    emotion_count: 1
    initial_baseline: 0.0
    relation_updates:
      shot_by: -0.6
    emotions:
      - name: calm
        initial_value: 0.0
```

의미: 어떤 좌석이 나를 쏘면(`shot_by`) 그 좌석에 대한 관계가 `-0.6`(원한)만큼 이동하고, `relation_target_weight = 1.5`이므로 그 좌석(인덱스 `s + 1`)을 쏘는 점수가 `-1.5 * (-0.6) = +0.9`만큼 올라갑니다. `relation_target_weight`를 기본값 `0.0`으로 두면 관계 항이 전혀 더해지지 않아 기존 동작과 동일합니다. 관계 벡터 작성 규칙은 [`dummy_ai_emotion_guide.md`](./dummy_ai_emotion_guide.md) 3.4/3.7 절을 보세요.

### 예시 C(코드 수정): 새 점수화 항 추가하기

표에 없는 새 성향을 만들려면 `intent_policy.py`를 편집해 **새 노브**를 추가해야 합니다. 예로 "생존 좌석이 적을수록(막판일수록) 공격을 더 공격적으로" 만드는 가상의 노브 `late_game_attack_weight`를 추가한다고 합시다. 세 곳을 함께 손봐야 합니다.

**(1) `IntentPolicyConfig`에 필드 추가** (`intent_policy.py`):

```python
@dataclass(frozen=True)
class IntentPolicyConfig:
    ...
    target_health_weight: float = 0.0
    relation_target_weight: float = 0.0
    late_game_attack_weight: float = 0.0  # 새 노브, 기본 0.0으로 하위 호환
    role_target_weights: Mapping[str, float] = field(default_factory=dict)
```

**(2) `__post_init__`의 유한성 검증 목록에 새 필드 등록** -- 그래야 무한대/NaN/비숫자 입력이 거부됩니다.

```python
        for name in (
            "pass_score",
            ...
            "target_health_weight",
            "relation_target_weight",
            "late_game_attack_weight",  # 새 노브도 유한성 검증
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
```

`from_value`는 `accepted = set(cls.__dataclass_fields__)`로 **데이터클래스 필드 집합을 자동으로** 받아들이므로, 필드를 추가하면 매핑/프리셋에서 자동으로 수용됩니다(별도 수정 불필요). 다만 다른 철자 별칭(예: `faction_tie_policy -> tie_policy`)을 지원하려면 `from_value`에 그 매핑을 추가하면 됩니다.

**(3) `IntentModule.decision`의 점수화에 새 항 반영** -- 예를 들어 생존 좌석 수가 적을 때 모든 대상 점수에 가중을 더하도록:

```python
        late_bonus = self.config.late_game_attack_weight * (PLAYER_COUNT - len(living))
        for target in living:
            if target == self.player_idx:
                continue
            target_scores[target] += late_bonus
```

마지막으로 `state_dict`/`from_state_dict`로 이 노브를 직렬화에 포함하고 싶다면 `IntentPolicyConfig.state_dict`에도 `"late_game_attack_weight": float(self.late_game_attack_weight)` 한 줄을 더하면 됩니다.

> 주의: 새 노브의 **기본값은 반드시 중립값(여기서는 `0.0`)** 으로 두세요. 그래야 기존 설정과 결정성(같은 시드 -> 같은 행동)이 유지됩니다. 그리고 의도 모듈이 추가하는 어떤 점수도 공통 행동 마스크를 넘어 불법 행동을 선택하게 만들 수 없습니다(마스크가 항상 최종 권위).

---

## 7. 빠른 체크리스트

- [ ] 성향 조절이 2.3 절 표의 **기존 노브**로 가능한가? 그렇다면 `config.preset.intent_policy`(= `config.example.yaml`의 `preset.intent_policy`) 값만 바꾼다(코드 수정 불필요).
- [ ] 특정 역할을 선호/기피하려면 `role_target_weights`에 `sheriff`/`deputy`/`outlaw_a`/`outlaw_b`/`traitor` 키로 가중치를 넣었는가?
- [ ] 원한 좌석을 겨냥하려면 `relation_target_weight`를 `0`보다 크게 두고, 감정 설정의 `relation_updates`로 관계를 쌓았는가? (기본 `0.0`이면 기존 동작과 동일)
- [ ] 새 점수화 항이 필요하면 `intent_policy.py`에서 (1) `IntentPolicyConfig` 필드 추가, (2) `__post_init__` 유한성 검증 등록, (3) `IntentModule.decision` 점수화 반영을 모두 했는가?
- [ ] 새 노브의 기본값을 중립값으로 두어 하위 호환과 결정성을 지켰는가?
- [ ] 모든 가중치가 유한 숫자인가? (무한대/NaN은 생성 시 거부됨)
