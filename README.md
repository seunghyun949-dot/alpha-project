# Alpha Project — EHD Venturi Simulator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seunghyun949-dot/alpha-project/blob/main/notebooks/ehd_venturi_colab.ipynb)

이 저장소는 알파프로젝트용 축대칭 EHD(Electrohydrodynamic) 벤츄리 시뮬레이터와 실험 데이터 기록/비교 도구를 관리합니다.

## Repository structure

```text
alpha-project/
├─ notebooks/
│  └─ ehd_venturi_colab.ipynb
├─ simulation/
│  ├─ ehd_venturi_colab_complete.py
│  └─ experiment_tools.py
├─ README.md
└─ requirements.txt
```

## Fastest way to run

README 상단의 **Open in Colab** 버튼을 누른 뒤 셀을 위에서부터 순서대로 실행하면 됩니다.

노트북은 GitHub의 최신 시뮬레이터와 실험 로거를 자동으로 내려받습니다.

## Simulator controls

- `r_in`: 입구 반지름
- `r_th`: throat 반지름
- `r_ou`: 출구 반지름
- `x_th`: throat 위치
- `u_inf`: 입구 유속
- `v_dis`: 방전 전극 전압
- `v_ind`: 유도 전극 전압
- `v_acc`: 가속 전극 전압
- `acc_offset`: 가속 전극 시작 위치
- `acc_length`: 가속 전극 길이

## Experiment logger

시뮬레이터를 실행한 뒤 `experiment_tools.py` 패널에서 다음 실측값을 입력할 수 있습니다.

- emitter type
- emitter diameter [mm]
- collector type
- electrode gap [mm]
- measured voltage [kV]
- measured current [mA]
- measured outlet velocity [m/s]

### Buttons

- **SAVE EXPERIMENT**: 현재 시뮬레이터 조건으로 다시 계산한 뒤 실측값과 함께 `experiment_results.csv`에 한 행 추가
- **SHOW COMPARISON**: 실측값과 시뮬레이션값 비교 그래프 출력
- **DOWNLOAD CSV**: 현재 실험 CSV를 PC로 다운로드
- **UPLOAD CSV**: 이전 실험 CSV를 업로드하여 이어서 기록

## CSV output

각 실험 행에는 다음과 같은 값이 함께 저장됩니다.

- test ID / timestamp
- emitter / collector 정보
- electrode gap
- measured voltage / current / velocity
- 현재 벤츄리 형상과 전극 전압 설정
- simulated force [mN]
- simulated throat velocity increment
- simulated throat / outlet velocity
- velocity error [m/s]
- velocity error [%]
- electrical power [W]
- model force per watt [mN/W]
- measured velocity gain

전력은 실측 `kV × mA = W` 관계로 계산합니다.

## Comparison graphs

현재 자동 비교 그래프는 다음 4개입니다.

1. measured outlet velocity vs simulated outlet velocity
2. measured voltage vs measured velocity
3. measured current vs measured velocity
4. test ID별 measured vs simulated velocity

또한 저장된 데이터가 있으면 평균/중앙 velocity error [%]를 출력합니다.

## Main simulator outputs

- Geometry + empirical charge density
- Electric potential
- Electric field magnitude
- EHD-only velocity increment
- Model body-force integral `Fx`
- throat 평균 `du`
- throat 평균 전체 유속
- outlet 평균 전체 유속

## Important limitation

현재 모델은 정량 검증된 코로나 방전 / 플라즈마-CFD / 실제 추력 예측 모델이 아닙니다.

특히 공간전하 `rho_e`는 전하수송 방정식을 직접 푸는 대신 전극 거리와 전기장 활성화를 이용한 경험적 모델입니다. 유동 역시 완전한 Navier–Stokes 해석이 아니라 벤츄리 면적비 기반 배경 유속과 단순 EHD 반응 모델을 사용합니다.

따라서 현재 출력되는 `Fx`, `du`, 유속 값은 절대 성능 예측값보다 실험 조건 간 상대 비교와 향후 모델 보정용으로 사용하는 것이 적절합니다.

## Planned improvements

- 공간전하 수송 방정식 추가
- corona onset 조건 반영
- ion mobility / diffusion 반영
- Poisson equation에 공간전하 항 결합
- 축대칭 Navier–Stokes 압력-속도 결합
- 실제 실험 데이터 기반 `rho_e`, `CHARGE_DECAY`, `EHD_RESPONSE_TIME` 보정
- 형상별/간격별 자동 회귀 및 실험 설계 분석
