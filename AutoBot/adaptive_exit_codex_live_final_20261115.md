# Codex 검토 원문 - 적응형 청산 엔진 실매매 연결 최종 확인 (2026-11-15)

- 도구: `codex exec --sandbox read-only`, 원문 그대로 보관
- 반영 결과: 설계 문서 §24

---

### 1. 확인된 결함 및 수정 판정

**이번 수정으로 새로 생긴 가동 차단 결함은 확인하지 못했습니다.**

- **항목 1 — 해결:** 보유 감소분과 진행 수량 중 작은 값만 `on_fill`에 반영하고, 미체결 잔량은 진행 중으로 유지합니다. 마감 시 `on_order_closed`를 호출하며, 체결이 없을 때만 단계를 되돌립니다. 근거: [r027_exit_live.py:348](D:/make_utils/Trading/real_trading_bot/AutoBot/live_trading/r027_exit_live.py:348), [마감 처리:356](D:/make_utils/Trading/real_trading_bot/AutoBot/live_trading/r027_exit_live.py:356).
- **항목 2 — 해결:** 첫 진입에서도 틱 파일을 읽고, REST 가격이 없어도 틱 가격이 있으면 포지션을 등록하여 STOP/FORCE 평가로 진행합니다. 가격이 전혀 없으면 대기합니다. 근거: [r027_exit_live.py:290](D:/make_utils/Trading/real_trading_bot/AutoBot/live_trading/r027_exit_live.py:290), [평가 경로:236](D:/make_utils/Trading/real_trading_bot/AutoBot/live_trading/r027_exit_live.py:236).

### 2. 가능한 회귀 위험

관찰: `restore_filled`와 복원 대기 마감은 체크포인트에 저장되지 않아, 대기 도중 다시 재시작하면 대기 시간이 새로 시작되고 이전 부분 체결 여부가 초기화됩니다. 근거: [r027_exit_live.py:109](D:/make_utils/Trading/real_trading_bot/AutoBot/live_trading/r027_exit_live.py:109), [초기화:107](D:/make_utils/Trading/real_trading_bot/AutoBot/live_trading/r027_exit_live.py:107).

### 3. 누락 테스트

관찰: 부분 체결 테스트는 잔량 유지와 STOP 수량까지 확인하지만, **부분 체결 후 마감 시 단계 유지**와 **첫 진입·stale REST에서 STOP 발생**은 직접 검증하지 않습니다. 근거: [test_r027_exit_live.py:224](D:/make_utils/Trading/real_trading_bot/AutoBot/tests/test_r027_exit_live.py:224), [237](D:/make_utils/Trading/real_trading_bot/AutoBot/tests/test_r027_exit_live.py:237).

### 4. 질문 / 전제

읽기 전용 정적 검토입니다. 파일을 수정하지 않았으며, 파일을 생성하는 테스트는 실행하지 않았습니다.

### 5. 검토 결론

**실매매 가동 가능** — 요청한 두 수정과 이번 변경으로 인한 가동 차단 결함에 한정한 판정입니다.