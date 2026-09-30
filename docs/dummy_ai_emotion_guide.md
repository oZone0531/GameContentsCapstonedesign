# 더미 AI 감정 설정 JSON 작성 가이드

이 문서는 더미 AI(DummyAgent)의 감정 설정 파일을 직접 작성하거나 편집하려는 사용자를 위한 안내서입니다. 감정 값이 어떤 조건에서 오르내리는지를 실제 실행 규칙(엔진 동작)에 맞춰 설명하고, 바로 저장/불러오기 가능한 예시를 제공합니다.

필드 이름과 JSON 키는 코드가 인식하는 리터럴이므로 영어 그대로 사용합니다. 설명은 한국어로 작성합니다.

> 참고: GUI 사용 흐름(설정 -> 더미 AI 설정, 파일로 저장/불러오기)은 `USER_GUIDE.md`에도 요약되어 있습니다. 이 문서는 그 파일의 JSON 구조와 작성 규칙을 더 깊게 다룹니다.

---

## 1. 핵심 개념: "한 파일에 모든 감정"

가장 흔한 오해부터 정리합니다.

- 감정 하나당 파일 하나가 아닙니다.
- **하나의 파일이 모든 감정을 담습니다.** 이 파일 전체가 `EmotionConfig` 하나에 대응합니다.
- 파일 안의 `emotions` 목록에 감정 정의(`EmotionDefinition`)가 여러 개 들어갑니다. 감정 6개를 쓰고 싶으면 `emotions` 목록에 6개의 항목을 넣습니다.

GUI의 "더미 AI 설정" 창에서 감정별로 보이는 행(row)과 그 안의 5개 JSON 입력 상자는, **이 단일 파일 안에 있는 한 감정(`EmotionDefinition`)의 다섯 개 규칙 맵**에 각각 대응합니다. 즉 GUI에서 감정 행을 추가하는 것은 파일의 `emotions` 목록에 항목을 하나 더 추가하는 것과 같습니다.

파일 형식은 JSON 또는 YAML(`.yml`/`.yaml`)을 지원합니다. JSON으로 내보낼 때는 사람이 읽기 좋게 들여쓰기 2칸에 키 정렬(sort_keys)된 형태로 저장됩니다.

---

## 2. 파일 최상위 구조: `EmotionConfig`

| 필드 | 타입 | 범위/검증 | 기본값과 동작 |
| --- | --- | --- | --- |
| `schema_version` | 문자열 | 공백 불가 | 기본 `"1.0"`. 스키마(감정 이름 구성)를 바꿔 다시 불러올 때 버전 관리용으로 사용됩니다. |
| `emotion_count` | 정수 | 1 이상 | 기본 `1`. 저장 시 실제 `emotions` 개수로 자동 보정됩니다. 값을 직접 맞출 필요는 없습니다. |
| `initial_baseline` | 실수 | `[0, 1]` | 기본 `0.0`. 어떤 감정의 `initial_value`가 `null`이면 그 감정의 초기값이자 회복(decay)이 향하는 기준선으로 사용됩니다. |
| `emotions` | 목록 | 아래 3절 참고 | 감정 정의 목록. 이름은 서로 **중복될 수 없습니다.** |

그 밖의 필드(`encoder`, `hidden_size=128`, `num_layers`, `batch_first`, `dropout`, `emotion_head_hidden_size`, `output_activation`, `loss`, `optimizer` 등)는 모두 고정 리터럴입니다. **생략하면 기본값이 그대로 쓰이므로 손으로 적을 필요가 없습니다.** 감정 설계에는 위 표의 4개 필드만 신경 쓰면 됩니다.

`emotions`는 다음 세 가지 형태로 적을 수 있습니다(불러올 때 자동 정규화됩니다).

1. 정의 객체의 목록(권장): `[{"name": "분노", ...}, {"name": "공포", ...}]`
2. 이름과 정의의 딕셔너리: `{"분노": {...}, "공포": {...}}`
3. 이름만 나열한 목록: `["분노", "공포"]` (나머지는 기본값)

---

## 3. 감정 하나의 정의: `EmotionDefinition`

| 필드 | 타입 | 범위/검증 | 기본값과 동작 |
| --- | --- | --- | --- |
| `name` | 문자열 | 비어 있지 않음, 감정 간 **고유** | 필수. 감정의 표시 이름입니다. |
| `initial_value` | 실수 또는 `null` | `[0, 1]` 또는 `null` | 기본 `null`. `null`이면 `initial_baseline`을 초기값으로 사용합니다. |
| `change_rate` | 유한 실수 | 유한값(음수 허용) | 기본 `0.0`. **이벤트에 대한 기본 변화량**입니다(3.1 절 참고). |
| `decay_rate` | 유한 실수 | 유한값 | 기본 `0.0`. 매 스텝 기준선으로 되돌아가는 회복 크기입니다. 음수는 `0`으로 취급됩니다. |
| `event_updates` | `{문자열: 유한숫자}` | 키는 비어 있지 않은 문자열, 값은 유한 숫자 | 기본 `{}`. 공개 이벤트 발생 시 변화 규칙. |
| `action_updates` | `{문자열: 유한숫자}` | 위와 동일 | 기본 `{}`. 유효한 게임 행동 시 변화 규칙. |
| `reward_updates` | `{문자열: 유한숫자}` | 위와 동일 | 기본 `{}`. 유효한 보상 발생 시 변화 규칙. |
| `terminal_updates` | `{문자열: 유한숫자}` | 위와 동일 | 기본 `{}`. 에피소드 종료 시 변화 규칙. |
| `effects` | `{문자열: 유한숫자}` | 위와 동일 | 기본 `{}`. 감정 값을 바꾸지 않고 행동 선택에 참고되는 자문(advisory)값. |

모든 감정 값은 **변화가 일어날 때마다 항상 `[0, 1]`로 잘립니다(clamp).** 규칙에서 큰 값을 줘도 1을 넘지 않고, 큰 음수를 줘도 0 아래로 내려가지 않습니다.

### 3.1 다섯 개 규칙 맵과 키 조회 순서

다섯 개 맵은 각각 서로 다른 사건에 반응합니다. 각 사건이 들어오면 엔진은 아래 순서대로 키를 찾아 **처음 발견된 키의 값**을 그 감정의 변화량으로 씁니다. `"*"`는 모든 경우를 받는 catch-all(포괄) 키입니다.

| 맵 | 반응 시점 | 키 조회 순서(위에서부터, 먼저 맞는 것 사용) | 아무 키도 없을 때 |
| --- | --- | --- | --- |
| `event_updates` | 공개 이벤트마다 | 1) `event_type` (소문자, 점으로 나뉜 마지막 조각) 2) 정규화 별칭(`elimination`->`player_eliminated`, `role_reveal`->`role_revealed`) 3) `"*"` | **`change_rate`만큼 변화** (기본 이벤트 변화량) |
| `action_updates` | 유효한 게임 행동마다(마스크가 있으면 허용된 행동에 한해) | 1) 행동 인덱스 문자열(`"0"`=pass, `"1"`=bang, 그 외 정수도 kind `bang`) 2) 행동 종류(`pass`/`bang`/`invalid`) 3) `"*"` | `0`(변화 없음) |
| `reward_updates` | 유효한 보상마다 | 1) 정확한 숫자 문자열(예: `"1"`, `"-0.5"`, 끝의 0 제거) 2) 부호 버킷(`positive`/`negative`/`zero`) 3) `"*"` | `0`(변화 없음) |
| `terminal_updates` | 에피소드 종료 시 | 1) 결과/결과명(소문자, 예: 팀·승패 문자열) 2) 생명주기 상태(`terminated`/`truncated`) 3) `normal`(정상 종료일 때만, truncated에는 안 붙음) 4) `"*"` | `0`(변화 없음) |
| `effects` | 행동 선택 시 조회(값 변화 없음) | 1) 행동 인덱스 키 2) 행동 종류 3) `action` 4) `"*"` | 해당 감정 항목 없음 |

중요한 점 몇 가지:

- `event_updates`의 기본값은 0이 아니라 **`change_rate`**입니다. 즉 특정 이벤트 키도 없고 `"*"`도 없으면, 그 감정은 이벤트가 올 때마다 `change_rate`만큼 움직입니다. 특정 이벤트에만 반응시키고 다른 이벤트에는 반응시키고 싶지 않다면 `change_rate`를 `0.0`으로 두고 원하는 이벤트 키만 명시하세요.
- `action_updates`/`reward_updates`/`terminal_updates`는 맞는 키가 없으면 변화가 0입니다. 항상 무언가 반응시키려면 `"*"`를 넣으세요.
- `effects`는 감정 값을 **직접 움직이지 않습니다.** 감정이 높거나 낮을 때 행동 선택에 참고되는 자문 신호로 노출됩니다(행동 선택 연동은 현재 보류 상태). 감정 자체를 올리고 싶으면 `effects`가 아니라 위 네 개 맵을 쓰세요.
- `decay_rate`는 매 환경 스텝마다 현재 값을 기준선(`initial_value` 또는 `initial_baseline`) 쪽으로 최대 `decay_rate`만큼 선형으로 끌어당깁니다. 즉 사건이 없을 때 감정이 서서히 평상시로 돌아오는 "회복 속도"입니다. 음수는 `0`으로 처리되어 회복이 없습니다.

### 3.2 엔진이 실제로 인식하는 키 vs. 예시 키

작성 시 혼동을 막기 위해 구분합니다.

- **엔진이 구조적으로 인식하는 키(철자 고정):**
  - 부호 버킷: `positive`, `negative`, `zero` (reward_updates)
  - 생명주기: `terminated`, `truncated`, `normal` (terminal_updates)
  - 행동 종류: `pass`, `bang`, `invalid`, 그리고 인덱스 `"0"`(pass) / `"1"`(bang) (action_updates, effects)
  - 포괄 키: `"*"` (모든 맵)
- **게임이 정의하는(가변) 키:** `event_updates`의 `event_type` 문자열과 `terminal_updates`의 결과명 문자열은 게임이 내보내는 값에 따라 달라집니다. 정확한 철자를 확신할 수 없다면 `"*"`(또는 이벤트의 경우 `change_rate`)로 안전하게 처리하고, 실제로 어떤 `event_type`이 나오는지 게임이 방출하는 이벤트로 확인한 뒤 구체 키를 추가하세요.

아래 예시에서 게임 정의 키는 "예시"라고 표시합니다.

---

## 4. 완성 예시 (a): 감정 6개짜리 전체 파일

바로 저장해서 불러올 수 있는 JSON입니다. 6개 감정(분노, 공포, 기쁨, 슬픔, 경계심, 자신감)을 정의합니다.

```json
{
  "schema_version": "1.0",
  "emotion_count": 6,
  "initial_baseline": 0.1,
  "emotions": [
    {
      "name": "분노",
      "initial_value": 0.0,
      "change_rate": 0.0,
      "decay_rate": 0.02,
      "event_updates": {
        "bang_hit": 0.15,
        "player_eliminated": 0.1,
        "*": 0.0
      },
      "action_updates": {
        "bang": 0.05,
        "pass": -0.02
      },
      "reward_updates": {
        "negative": 0.1,
        "positive": -0.05
      },
      "terminal_updates": {
        "truncated": 0.2,
        "terminated": -0.1
      },
      "effects": {
        "bang": 0.3,
        "*": 0.1
      }
    },
    {
      "name": "공포",
      "initial_value": 0.2,
      "change_rate": 0.0,
      "decay_rate": 0.03,
      "event_updates": {
        "player_eliminated": 0.2,
        "*": 0.01
      },
      "action_updates": {
        "invalid": 0.1
      },
      "reward_updates": {
        "negative": 0.15,
        "*": 0.0
      },
      "terminal_updates": {
        "terminated": 0.1,
        "*": 0.0
      },
      "effects": {
        "pass": 0.4
      }
    },
    {
      "name": "기쁨",
      "initial_value": null,
      "change_rate": 0.0,
      "decay_rate": 0.02,
      "event_updates": {
        "bang_hit": 0.1
      },
      "action_updates": {
        "bang": 0.05
      },
      "reward_updates": {
        "positive": 0.2,
        "negative": -0.1
      },
      "terminal_updates": {
        "normal": 0.3
      },
      "effects": {}
    },
    {
      "name": "슬픔",
      "initial_value": null,
      "change_rate": 0.0,
      "decay_rate": 0.01,
      "event_updates": {
        "player_eliminated": 0.15
      },
      "action_updates": {},
      "reward_updates": {
        "negative": 0.2,
        "zero": 0.02
      },
      "terminal_updates": {
        "terminated": 0.25,
        "normal": -0.1
      },
      "effects": {}
    },
    {
      "name": "경계심",
      "initial_value": 0.3,
      "change_rate": 0.01,
      "decay_rate": 0.02,
      "event_updates": {
        "role_revealed": 0.2
      },
      "action_updates": {
        "*": 0.0
      },
      "reward_updates": {
        "*": 0.0
      },
      "terminal_updates": {
        "*": 0.0
      },
      "effects": {
        "*": 0.2
      }
    },
    {
      "name": "자신감",
      "initial_value": 0.4,
      "change_rate": 0.0,
      "decay_rate": 0.02,
      "event_updates": {
        "bang_hit": 0.12,
        "*": 0.0
      },
      "action_updates": {
        "1": 0.08,
        "0": -0.03
      },
      "reward_updates": {
        "positive": 0.15,
        "negative": -0.1
      },
      "terminal_updates": {
        "normal": 0.2,
        "truncated": -0.05
      },
      "effects": {
        "bang": 0.5
      }
    }
  ]
}
```

이 예시에서 유의할 점:

- `initial_baseline`이 `0.1`이므로, `initial_value`가 `null`인 "기쁨"과 "슬픔"은 초기값 `0.1`에서 시작하고 회복도 `0.1`을 향합니다.
- "경계심"은 `change_rate`가 `0.01`이므로, `event_updates`에 없는 이벤트가 들어오면 매번 `+0.01`씩 오릅니다. 반면 다른 감정들은 `change_rate`가 `0.0`이거나 `"*": 0.0`을 두어 미지정 이벤트에는 반응하지 않습니다.
- `bang_hit`, `player_eliminated`, `role_revealed`는 게임이 내보내는 `event_type`을 가정한 **예시 키**입니다. 실제 방출 철자를 확인한 뒤 조정하세요.
- 결과명 기반 `terminal_updates` 키(예: 특정 팀 승리명)는 게임 정의 값이므로 위 예시는 안전하게 `normal`/`terminated`/`truncated`/`"*"` 같은 구조적 키만 사용했습니다.

---

## 5. 완성 예시 (b): 감정 하나("분노") 자세히 따라가기

GUI의 "더미 AI 설정" 창에서 "분노" 감정 행에 무엇을 입력하는지를 파일 관점과 나란히 봅니다. 이 행 하나가 위 파일의 `emotions` 목록 안 항목 하나가 됩니다.

GUI 입력 항목과 파일 필드 대응:

| GUI 입력 | 파일 필드 | 이 예시에서 넣을 값 |
| --- | --- | --- |
| 감정 이름 | `name` | `분노` |
| 초기값 | `initial_value` | `0.0` |
| 변화율(기본 이벤트 변화량) | `change_rate` | `0.0` |
| 회복율 | `decay_rate` | `0.02` |
| 이벤트 규칙 상자(JSON) | `event_updates` | 아래 |
| 행동 규칙 상자(JSON) | `action_updates` | 아래 |
| 보상 규칙 상자(JSON) | `reward_updates` | 아래 |
| 종료 규칙 상자(JSON) | `terminal_updates` | 아래 |
| 효과 상자(JSON) | `effects` | 아래 |

다섯 개 JSON 상자에 각각 다음을 입력합니다.

이벤트 규칙 상자(`event_updates`):

```json
{
  "bang_hit": 0.15,
  "player_eliminated": 0.1,
  "*": 0.0
}
```

의미: `bang_hit` 이벤트가 오면 분노 `+0.15`, `player_eliminated`(또는 별칭 `elimination`)면 `+0.1`, 그 외 이벤트는 `"*": 0.0`이 잡아 변화 없음. `"*"`가 있으므로 `change_rate`는 사용되지 않습니다.

행동 규칙 상자(`action_updates`):

```json
{
  "bang": 0.05,
  "pass": -0.02
}
```

의미: 유효한 BANG 행동(`"1"` 또는 종류 `bang`)이면 `+0.05`, PASS(`"0"` 또는 종류 `pass`)면 `-0.02`. `"*"`가 없으니 그 밖의 유효 행동은 변화 없음.

보상 규칙 상자(`reward_updates`):

```json
{
  "negative": 0.1,
  "positive": -0.05
}
```

의미: 음수 보상이면 분노 `+0.1`, 양수 보상이면 `-0.05`, 0 보상은 키가 없으니 변화 없음.

종료 규칙 상자(`terminal_updates`):

```json
{
  "truncated": 0.2,
  "terminated": -0.1
}
```

의미: 무승부(truncated, 시간 초과)로 끝나면 분노 `+0.2`, 정상 종료(terminated)면 `-0.1`. 결과명 키를 쓰지 않았으므로 구조적 상태 키만으로 동작합니다.

효과 상자(`effects`):

```json
{
  "bang": 0.3,
  "*": 0.1
}
```

의미: BANG 계열 행동에는 자문값 `0.3`, 그 외에는 `0.1`. 이 값은 분노 수치를 직접 바꾸지 않고 행동 선택 참고용으로만 노출됩니다.

이 감정 행 하나를 저장하면 파일의 `emotions` 배열에 다음 항목으로 합쳐집니다.

```json
{
  "name": "분노",
  "initial_value": 0.0,
  "change_rate": 0.0,
  "decay_rate": 0.02,
  "event_updates": { "bang_hit": 0.15, "player_eliminated": 0.1, "*": 0.0 },
  "action_updates": { "bang": 0.05, "pass": -0.02 },
  "reward_updates": { "negative": 0.1, "positive": -0.05 },
  "terminal_updates": { "truncated": 0.2, "terminated": -0.1 },
  "effects": { "bang": 0.3, "*": 0.1 }
}
```

동작을 시간순으로 상상해 보면:

1. 게임 중 `bang_hit` 이벤트 발생 -> 분노 `0.0 -> 0.15`.
2. 사건이 없는 스텝이 지나감 -> `decay_rate=0.02`만큼 기준선(`initial_value=0.0`) 쪽으로 회복 -> `0.15 -> 0.13`.
3. 음수 보상 발생 -> `+0.1` -> `0.13 -> 0.23`.
4. 여러 사건이 겹쳐 값이 커져도 항상 `[0, 1]`로 잘립니다.

---

## 6. 작성자 친화 별칭(alias)

불러올 때 아래 별칭도 정식 필드로 자동 변환됩니다. 손으로 쓸 때 편한 이름을 써도 됩니다.

| 별칭 | 정식 필드 |
| --- | --- |
| `initial`, `baseline` | `initial_value` |
| `change`, `delta` | `change_rate` |
| `decay` | `decay_rate` |
| `event_rules` | `event_updates` |
| `action_rules` | `action_updates` |
| `reward_rules` | `reward_updates` |
| `terminal_rules` | `terminal_updates` |
| `action_effects` | `effects` |

예를 들어 아래 두 표기는 동일하게 해석됩니다.

```json
{ "name": "공포", "baseline": 0.2, "decay": 0.03, "event_rules": { "*": 0.01 } }
```

```json
{ "name": "공포", "initial_value": 0.2, "decay_rate": 0.03, "event_updates": { "*": 0.01 } }
```

---

## 7. 저장, 불러오기, 검증

- GUI에서 **설정 -> 더미 AI 설정 -> 감정 설정 파일로 저장**을 누르면 현재 감정 설정 전체가 하나의 파일(JSON 또는 YAML)로 내보내집니다.
- **감정 설정 파일 불러오기**를 누르면 저장했거나 손으로 편집한 파일을 다시 읽어 검증 후 적용합니다.
- 이 파일은 GUI 밖에서 자유롭게 편집할 수 있습니다. 텍스트 편집기로 열어 값만 바꿔 저장해도 됩니다.
- 불러오기 시 다음과 같은 검증 오류가 화면에 표시됩니다.
  - 감정 이름 중복(`name`은 서로 달라야 함)
  - `initial_value` 또는 `initial_baseline`이 `[0, 1]`를 벗어남
  - 유한하지 않은 숫자(무한대/NaN)나 규칙 값이 숫자가 아님
  - 규칙 맵의 키가 빈 문자열

문제가 없으면 값이 검증된 뒤 화면과 실행에 반영됩니다.

---

## 8. 빠른 체크리스트

- [ ] 감정마다 `name`이 고유한가?
- [ ] 특정 이벤트에만 반응시키고 싶다면 `change_rate=0.0`으로 두고 원하는 이벤트 키만 넣었는가? (그렇지 않으면 미지정 이벤트가 `change_rate`만큼 움직임)
- [ ] `action`/`reward`/`terminal` 맵에서 항상 반응이 필요하면 `"*"`를 넣었는가?
- [ ] `effects`로 감정 값을 직접 올리려 하지 않았는가? (값 변화는 앞 네 개 맵에서만)
- [ ] 게임 정의 키(`event_type`, 결과명)의 철자를 확신하는가? 확신이 없으면 `"*"` 또는 `change_rate`로 처리했는가?
- [ ] 모든 숫자가 유한값이고 초기값이 `[0, 1]` 안에 있는가?
