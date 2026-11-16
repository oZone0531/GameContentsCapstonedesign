# 더미 AI 감정(9감정 공식) 설정 가이드

이 문서는 더미 AI(DummyAgent)의 감정 설정 파일을 직접 작성하거나 편집하려는 사용자를 위한 안내서입니다. 더미 AI는 **항상 9개의 감정 값**으로만 행동하며, 이 값들은 감정 공식(`bang_rl/agents/dummy_agent/emotion_formula.py`의 `FormulaEmotionModule`)이 공개 사건으로부터 계산합니다. 사용자가 조정하는 것은 감정의 개수나 규칙이 아니라 **공식의 계수(`formula`)와 감정 설명(`description`)** 입니다.

필드 이름과 JSON 키는 코드가 인식하는 리터럴이므로 영어 그대로 사용합니다. 설명은 한국어로 작성합니다.

> 참고: GUI 사용 흐름(설정 -> 더미 AI 설정, 파일로 저장/불러오기)은 `USER_GUIDE.md`에도 요약되어 있습니다. 이 문서는 그 파일의 JSON 구조와 공식 계수를 더 깊게 다룹니다.

---

## 1. 9개의 감정

감정 라벨은 아래 순서로 길이 9, 각 값은 `[0, 1]` 입니다(`LABEL_NAMES`).

| 순서 | 이름 | 의미 |
| --- | --- | --- |
| 0 | `calm` | 파생값: `1 - max(confidence, aggression, anxiety)` |
| 1 | `confidence` | 내 편에게 좋은 일(적이 맞거나 탈락)이 생기면 오른다 |
| 2 | `aggression` | 통제감이 남아 있는 상태에서 나쁜 일을 겪은 비율(HP 기반 통제 평가) |
| 3 | `anxiety` | 통제감이 없는 상태에서 나쁜 일을 겪은 비율(HP 기반 통제 평가) |
| 4~8 | `relation_0` ~ `relation_4` | 좌석 0~4에 대한 관계. 내부 값 `R`은 `[-1, 1]`(음수 = 원한, 양수 = 호감)이고 라벨에는 `(R + 1) / 2`로 저장된다(0.5 = 중립). 자기 자신 좌석은 항상 중립 |

감정은 공개 정보(공개 사건 `bang`/`player_eliminated`, 공개 체력과 생존, 역할 믿음)에서만 계산됩니다. 비공개 역할은 사용하지 않습니다.

---

## 2. 파일 구조: `EmotionConfig`

하나의 파일이 `EmotionConfig` 하나에 대응합니다. 파일 형식은 JSON 또는 YAML(`.yml`/`.yaml`)입니다. 모든 필드는 생략할 수 있으며, 생략하면 9감정 + 튜닝된 공식이 쓰입니다.

| 필드 | 타입 | 기본값과 동작 |
| --- | --- | --- |
| `schema_version` | 문자열(공백 불가) | 기본 `"formula-1.0+label9"`. 데이터셋과 메인 AI 감정 추론 모듈이 같은 감정 스키마인지 확인하는 데 쓰입니다. |
| `emotions` | 목록 | 기본은 위 9개 이름. 항목은 `name`(필수, 중복 불가)과 `description`(선택)을 가집니다. 문자열 목록(`["calm", ...]`)이나 이름-정의 딕셔너리로도 적을 수 있습니다. |
| `emotion_count` | 정수 | `emotions` 길이로 자동 결정됩니다. 직접 맞출 필요가 없습니다. |
| `initial_baseline` | 실수 `[0, 1]` | 기본 `0.0`. 메인 AI 감정 추론 모듈의 초기 기준값입니다. 더미 AI 계산에는 쓰이지 않습니다. |
| `formula` | 객체 | 감정 계산식 블록. 생략하면 저장소 맨 위 `emotion_formula.example.json`의 튜닝된 공식이 쓰입니다. 알 수 없는 이름은 오류입니다. |

`description`은 관전 모드와 리플레이에서 감정 막대에 마우스를 올리면 툴팁으로 표시되는 자유 서술입니다. 작성자가 쓴 글 그대로 저장되며 비워 둬도 됩니다.

그 밖의 필드(`encoder`, `hidden_size=128`, `num_layers`, `batch_first`, `dropout`, `emotion_head_hidden_size`, `output_activation`, `loss`, `optimizer`)는 메인 AI 감정 추론 신경망의 고정 리터럴이므로 손으로 적을 필요가 없습니다.

---

## 3. `formula` 블록

```json
"formula": {
  "schema_version": "formula-1.0",
  "params": { "theta": 0.8, "kappa_F": 3.0 },
  "personality_by_role": { "outlaw": { "b_A": 0.3 } }
}
```

- `params`: 아래 3.1 표의 계수 중 바꿀 것만 적습니다. 적지 않은 계수는 코드 기본값입니다. 값은 유한 숫자여야 하며, 표에 없는 이름은 오류입니다.
- `personality_by_role`: 키는 `sheriff` / `deputy` / `outlaw` / `traitor`. 그 역할을 맡은 더미에게만 `params`를 덮어쓰는 성격 값입니다. 다른 키는 오류입니다.

### 3.1 계수 표 (`EmotionParams`)

| 묶음 | 계수 | 코드 기본값 | 의미 |
| --- | --- | --- | --- |
| 사건 크기 | `beta_bang` | 0.15 | bang 한 번의 기본 크기 |
| | `beta_elim` | 0.25 | 탈락 시 bang 위에 더해지는 크기 |
| | `nu_hp` | 0.0 | 대상 체력이 낮을 때 더 크게 반응(0이면 끔) |
| 좋은/나쁜 사건 가중 | `omega_self` | 1.0 | 내가 맞음 |
| | `omega_ally` | 0.6 | 아군이 맞음(아군일 확률을 곱함) |
| | `omega_enemy` | 0.5 | 적이 맞음(적일 확률을 곱함) |
| 상호 억제 | `lambda_S` | 0.3 | 나쁜 사건이 자신감을 낮추는 정도 |
| | `lambda_A` | 0.0 | 좋은 사건이 공격성을 낮추는 정도 |
| | `lambda_F` | 0.3 | 좋은 사건이 불안을 낮추는 정도 |
| 통제 평가 | `w_hp` | 0.5 | 통제감에서 내 체력의 비중(나머지는 팀 체력) |
| | `theta` | 0.5 | 공격성/불안을 가르는 문턱 |
| | `k` | 8.0 | 문턱의 가파름 |
| 관계 | `eta_hurt` | 1.0 | 나를 쏜 상대에 대한 원한 |
| | `eta_ally` | 0.5 | 아군을 쏜 상대에 대한 원한 |
| | `eta_avenge` | 0.6 | 내가 원한을 품은 상대를 쏜 사람에 대한 호감 |
| | `eta_enemy` | 0.0 | (적일 가능성이 높은) 상대를 쏜 사람에 대한 호감(0이면 끔) |
| | `rho_B` | 0.0 | 관계가 역할 믿음 목표를 따라가는 속도(0이면 끔) |
| | `g_B` | 2.0 | 상대적 아군 확률을 믿음 목표로 바꾸는 이득 |
| | `traitor_sheriff_ally` | 1.0 | 배신자가 보안관을 "살려 둘 대상"으로 보는 정도(1:1 전까지) |
| 성격 | `p_S` `p_A` `p_F` `p_R` | 1.0 | 자신감/공격성/불안/관계의 민감도 |
| | `b_S` `b_A` `b_F` | 0.1 | 자신감/공격성/불안의 기준선 |
| | `d_S` `d_A` `d_F` | 0.01 | 자신감/공격성/불안의 회복 속도 |
| | `d_R` | 0.005 | 관계의 회복 속도 |
| 잡음 | `sigma` | 0.01 | 자신감/공격성/불안의 스텝당 잡음 표준편차 |
| | `sigma_R` | 0.005 | 관계의 스텝당 잡음 표준편차 |
| 행동 기여 | `kappa_S` `kappa_A` `kappa_F` `kappa_R` | 1.0 | 감정별 행동 점수 기여 크기 |
| | `kappa_AR` | 0.0 | 원한 상대를 겨냥한 추가 bang 점수(공격성 x 원한, 0이면 끔) |
| | `lambda_E` | 1.0 | 감정 기여분 전체의 배율(0이면 의도 점수만으로 행동) |

저장소의 `emotion_formula.example.json`에는 튜닝된 값이 들어 있습니다(예: `theta=0.8`, `kappa_F=3.0`). 위 표의 "코드 기본값"은 `params`에 적지 않았을 때의 값이고, `EmotionConfig()`의 기본 `formula`는 이 튜닝된 값입니다.

---

## 4. 감정이 행동에 미치는 영향

`FormulaEmotionModule.action_contribution()`이 길이 6 벡터(인덱스 0 = pass, 1~5 = 좌석 `s = 인덱스 - 1`을 쏘는 bang)를 만들고, 의도 점수에 더해져 결합 점수가 됩니다. 불법 행동은 공통 행동 마스크가 항상 배제하므로 감정 계수를 크게 잡아도 불법 행동이 선택되지 않습니다. 감정 잡음의 난수도 에이전트 시드에서 파생되므로 같은 시드는 같은 행동을 만듭니다.

살아 있는 모든 더미는 자기 차례가 아닐 때도 스텝마다 자신의 공개 관측으로 감정을 갱신합니다(`observe_public_boundary`).

---

## 5. 사용하기

- **GUI:** 설정 -> 더미 AI 설정에서 감정 설명과 `formula` JSON을 편집하고 파일로 저장/불러오기를 할 수 있습니다.
- **데이터 수집:** `python collect_data.py --games 1000 --out <폴더> [--emotion-file my_formula.json]`. 파일을 주지 않으면 `emotion_formula.example.json`을 씁니다. 각 스텝에는 살아 있는 더미마다 9개 값의 감정 라벨이 기록됩니다.
- **측정과 추적:** `python measure_emotion.py --emotion-file my_formula.json`, `python trace_emotion.py`로 감정이 행동을 얼마나 바꾸는지 확인합니다.

감정 라벨의 스키마 이름은 `formula-1.0+label9` 이며, 메인 AI의 감정 추론 모듈도 같은 9개 이름을 사용해야 합니다.
