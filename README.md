# Alpha Project — EHD Venturi Simulator

이 저장소는 알파프로젝트용 축대칭 EHD(Electrohydrodynamic) 벤츄리 시뮬레이터 코드를 관리합니다.

현재 구현은 Google Colab에서 실행하는 것을 기준으로 하며, `ipywidgets` 기반 UI를 사용해 벤츄리 형상과 전극 전압/위치를 조절한 뒤 결과를 다시 계산할 수 있습니다.

## Repository structure

```text
alpha-project/
├─ simulation/
│  └─ ehd_venturi_colab_complete.py
├─ README.md
└─ requirements.txt
```

## Main simulator

`simulation/ehd_venturi_colab_complete.py`

현재 시뮬레이터는 다음을 계산/표시합니다.

- 축대칭 벤츄리 형상
- 방전 전극 / 유도 전극 / 가속 전극 배치
- 축대칭 Laplace 전위 해석
- 전기장 크기
- 경험적 공간전하 분포
- Coulomb 체적력 `f = rho_e E`
- 벤츄리 면적비 기반 배경 유속
- 단순화된 EHD 유속 증가량 `du`
- 축대칭 체적 적분 기반 모델 힘 `Fx`

## Google Colab 실행

Colab 첫 셀에서:

```python
%pip install -q numpy scipy matplotlib ipywidgets
%matplotlib inline
```

그 다음 `simulation/ehd_venturi_colab_complete.py` 전체를 실행하거나, 저장소를 Colab에 불러와 `%run`으로 실행하면 됩니다.

예시:

```python
%run simulation/ehd_venturi_colab_complete.py
```

UI가 표시되면 슬라이더 값을 조절하고 `RECALCULATE` 버튼을 눌러 새 결과를 계산합니다.

## Main controls

- `r_in`: 입구 반지름
- `r_th`: throat 반지름
- `r_ou`: 출구 반지름
- `x_th`: throat 위치
- `u_inf`: 입구 유속
- `v_dis`: 방전 전극 전압
- `v_ind`: 유도 전극 전압
- `v_acc`: 가속 전극 전압
- `acc_offset`: throat 기준 가속 전극 시작 위치
- `acc_length`: 가속 전극 길이

## Outputs

주요 출력은 다음과 같습니다.

1. Geometry + empirical charge density
2. Electric potential
3. Electric field magnitude
4. EHD-only velocity increment
5. Model body-force integral `Fx`
6. throat 평균 `du`
7. throat 평균 전체 유속
8. outlet 평균 전체 유속

## Important limitation

현재 모델은 정량 검증된 코로나 방전 / 플라즈마-CFD / 실제 추력 예측 모델이 아닙니다.

특히 공간전하 `rho_e`는 전하수송 방정식을 직접 푸는 대신 전극으로부터의 거리와 전기장 활성화 계수를 이용한 경험적 모델입니다. 유동 역시 완전한 Navier–Stokes 해석이 아니라 벤츄리 면적비 기반 배경 유속과 `fx / rho * response_time` 형태의 단순 EHD 반응 모델을 사용합니다.

따라서 현재 출력되는 `Fx`, `du`, 유속 값은 실험 추력의 절대값으로 사용하기보다 전극 위치, 형상, 전압 변화에 따른 상대 비교 지표로 사용하는 것이 적절합니다.

## Planned improvements

- 공간전하 수송 방정식 추가
- corona onset 조건 반영
- ion mobility / diffusion 반영
- Poisson equation에 공간전하 항 결합
- 축대칭 Navier–Stokes 압력-속도 결합
- 실제 실험 데이터 기반 `rho_e` 및 유동 모델 보정
- 전극 형상 세분화
- 실험 결과 CSV와 시뮬레이션 결과 비교 자동화
