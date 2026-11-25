/*
 * Update log
 * - [2026-11-05] type=feat - Control tab (guide 2.1): r003 service state (running/PID/start time/restarts/mode/profile), risk
 *   profile + dry-run choice, 실행/정지/재시작 with the guide's enable rules (start = not running and no external r003), a spinner
 *   while the main process waits for the unit to settle, the external-r003 warning, pending-restart items (overrides with
 *   [기본값으로], env, watchlist), override errors, and the last 20 history rows (`history -n 20`). Confirmations (live orders,
 *   market hours) are shown by the main process. Server times are shown as the server's KST strings, unconverted
 * - [2026-11-07] type=fix - Mode label: status.dry_run is null while the bot is stopped (no running snapshot); it was shown as a red
 *   '실주문'. Now '-' when stopped, '확인 불가' when running without a snapshot, dry-run/실주문 only from the snapshot
 * - [2026-11-10] type=feat - Local mode: host line (hostname, systemd / r020 supervisor, host role primary|standby with a switch,
 *   tick collector state + 시작), PeerPanel for the other host (status, its r003 stop, standby/primary, watchlist copy), local
 *   wording for the external-r003 warning and the restart note
 * - [2026-11-11] type=feat - "매매 실행 위치" section on top (HostPanel): this PC and the Raspberry Pi as two rows (role, r003, tick
 *   collector), the selected row is the host [실행] starts r003 on; [여기서 실행] on the other row saves runMode and reconnects
 *   (no trip to 설정/연결). HostLine + PeerPanel merged into it; role / collector / peer stop / watchlist copy moved under "고급".
 *   The start button says where it starts ("이 PC에서 실행" / "라즈베리파이에서 실행")
 * - [2026-11-11] type=fix - Host row: tick collector shows its symbol count and "구독 누락 n종목" (watch codes it does not collect)
 *   with a [수집기 재시작] button on the selected host
 * - [2026-11-11] type=feat - Card layout (reference design): every block is a card with a head row (title, muted subtitle / count,
 *   the actions on the right). 매매 실행 위치 = radio rows (host, role, r003 state, tick collector); 봇 상태 = buttons in the head
 *   ([재시작] [정지] [<host>에서 실행]), a 4-cell stat strip (상태 / 모드 / 시작 / PID · 재시작), then the profile row with the
 *   external-r003 / restart note on the right; 미적용 변경 and 최근 변경 이력 as cards. [정지] is red while r003 runs
 * - [2026-11-11] type=feat - "!" next to the profile select (ProfileInfo): one-line summary of conservative / neutral / aggressive
 *   and a table of the keys each profile changes (server risk_profiles/*.json), plus what stays the same in all of them
 * - [2026-11-18] type=perf - Peer status, change history, override list and Edge AI model: the last answers (dataCache) are shown
 *   at once when the tab is opened again, then refreshed
 * - [2026-11-25] type=feat - layout (user 2026-11-25): dry-run right beside the profile's "!" (the description opens below); 매매 방식 as
 *   a box with one block per feature - 규칙 매매 / Edge AI 매매 (sub-options indented: 가상 매수 (locked on - real orders are a later
 *   server feature), 신호 후보만 기록, 청산 기준) / 반등 알림 - and a separate 매도 설정 group (자동 매도, 자동 매도 대상)
 * - [2026-11-25] type=feat - layout (user): 청산 기준 on top (not under Edge AI), 매매 방식 = 규칙 매매 / Edge AI 매매 with the sub
 *   checkboxes 실제 매매 (EDGE_LIVE_ORDERS) / 가상 매매 (the opposite of EDGE_ONLY_LOG_ONLY; "신호 후보만 기록" is gone - both
 *   unchecked = candidates only), 반등 알림 as its own group
 * - [2026-11-25] type=feat - Edge AI 매매 > "가상 매수" can be unchecked (server EDGE_LIVE_ORDERS): Edge AI live orders beside the
 *   rule strategy. Its limits (1회 금액 / 동시 보유 / 하루 횟수 / 하루 손실 한도, EdgeLiveLimitsEdit) and today's edge summary
 *   (r016 status edge_live) appear below; needs 자동 매도 = 전체 (the server refuses the other combinations); an older server
 *   keeps the checkbox locked
 * - [2026-11-25] type=feat - 매매 방식 = checkboxes 규칙 매매 (server RULE_BUY_ENABLED) / Edge AI 가상 매매 (EDGE_ONLY_MODE) / 반등 알림
 *   (AutoBotHtsH only, found through window.api.reboundStatus / reboundEnable) in any combination (user 2026-11-25, Codex review).
 *   The checkboxes never change TRADING_STRATEGY_MODE; a server on edge / basic shows it with a 규칙 매매 방식으로 전환 button
 * - [2026-11-25] type=feat - the exits box is "Edge AI 청산": one place for the paper runner and the edge strategy's real sells
 *   (server r006 reads the same keys), shown in paper and edge mode; empty box = the model (paper) / the server default (edge)
 * - [2026-11-25] type=feat - 가상 청산 익절 / 손절 / 보유 시간 boxes beside 가상 매수 실행 (server r001 EDGE_ONLY_TP_PCT /
 *   EDGE_ONLY_SL_PCT / EDGE_ONLY_HORIZON_MIN, empty = the model's label shown as the placeholder; r003 restart, new paper buys only)
 */

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { AUTO_SELL_KEY, AUTO_SELL_KO, AUTO_SELL_MODES, AUTO_SELL_SCOPE_KEY, AUTO_SELL_SCOPE_KO, AUTO_SELL_SCOPES, autoSellOf, autoSellScopeOf, type AutoSellScope, edgeLabelKo, TRADE_MODE_KO, tradeModeChanges, tradeModeOf, PAPER_LOG_ONLY_KEY, type AutoSellMode, type EdgeModelInfo, type TradeMode, PAPER_EXIT_KINDS, paperExitChanges, paperExitInput, type PaperExitKind, type PaperExitState, comboChanges, comboLabel, TRADE_KIND_KO, type TradeKind, EDGE_LIVE_KEY, EDGE_LIVE_ON_WARNING, EDGE_LIVE_EXIT_NOTE, EDGE_LIVE_LIMIT_KINDS, EDGE_LIVE_LIMIT_KO, EDGE_LIVE_LIMIT_RANGE, EDGE_LIVE_LIMIT_UNIT, edgeLiveLimitChanges, edgeLiveLimitInput, edgeLiveSummaryText, parseEdgeLiveSummary, type EdgeLiveLimitKind, type EdgeLiveLimits } from '../../../shared/paper';
import { useStore } from '../store';
import { peek, put } from '../dataCache';
import type { RunMode } from '../../../shared/types';
import {
  botSummary,
  fmtValue,
  historyText,
  pctHint,
  pendingCount,
  RISK_PROFILES,
  serviceButtons,
  type HistoryRow,
  type R016Status,
  type RiskProfile,
  type SchemaKey,
} from '../../../shared/control';

const isoShow = (s: string | null | undefined) => (s ? s.replace('T', ' ').slice(0, 19) : '-');
/** "HH:MM:SS" of a server KST string (the date stays in the tooltip) */
const timeOnly = (s: string | null | undefined) => (s ? isoShow(s).slice(11) || '-' : '-');

export const BOT_LABEL: Record<string, string> = {
  unknown: '상태 모름',
  running: '실행 중',
  starting: '시작 중',
  stopping: '정지 중',
  stopped: '정지',
  failed: '실패(정지)',
  external: '서비스 밖에서 실행 중',
};

const BUSY_LABEL: Record<string, string> = {
  start: '실행 중… 서비스가 올라올 때까지 확인합니다',
  stop: '정지 중… r003이 상태를 저장하고 내려갈 때까지 최대 90초 걸릴 수 있습니다',
  restart: '재시작 중… 정지(최대 90초) 후 다시 시작합니다',
};

const ACTION_LABEL: Record<string, string> = {
  set: '설정',
  unset: '기본값',
  reset: '전체 되돌림',
  env: '실행 환경',
  watchlist: '감시목록',
  'service.start': '실행',
  'service.stop': '정지',
  'service.restart': '재시작',
  role: '호스트 역할',
  'collector.start': '틱 수집기 시작',
  'collector.restart': '틱 수집기 재시작',
};

const roleLabel = (r: string | undefined) => (r === 'standby' ? 'standby (매매 안 함)' : 'primary (매매 호스트)');

/**
 * [2026-11-11] What the three risk profiles change (server live_trading/risk_profiles/<name>.json, applied over r001 when r003
 * starts - only keys the live code reads are listed; TAKE_PROFIT_PERCENT / STOP_LOSS_* in those files are legacy and unused).
 * Hard stop (-3.5% after 4 min), ATR stop (1.5 x ATR, never closer than -3.5%), TP1 (+3.0%, 40%) and profit lock are the same in every profile.
 */
const PROFILE_SUMMARY: Record<RiskProfile, string> = {
  conservative: '보수 - 종목당 30만원, 작은 반납에도 빨리 정리',
  neutral: '중립 - 종목당 50만원, 기본값 (현재 실전 기준)',
  aggressive: '공격 - 종목당 70만원, 되돌림을 더 참고 추세를 길게 탐',
};
const PROFILE_ROWS: { k: string; v: Record<RiskProfile, string>; hint?: string }[] = [
  { k: '종목당 주문 금액 상한', v: { conservative: '30만원', neutral: '50만원', aggressive: '70만원' } },
  { k: '익절 후 트레일 (1차 익절 뒤 잔량, 고점 대비)', v: { conservative: '사용 안 함', neutral: '-1.0%', aggressive: '-0.7%' }, hint: 'ENABLE_TP_EXTENSION_TRAILING / TP_EXTENSION_TRAIL_FROM_PEAK' },
  { k: '트레일링 스탑 (고점 대비 반납)', v: { conservative: '-1.2%', neutral: '-1.5%', aggressive: '-2.8%' }, hint: 'TRAILING_STOP_FROM_PEAK' },
  { k: '매수 직후 급락 가드 (0~4분)', v: { conservative: '-3.5% · 25초', neutral: '-3.5% · 60초', aggressive: '-3.5% · 60초' }, hint: 'POST_BUY_BB_DROP_PCT / POST_BUY_DROP_CONFIRM_SECONDS' },
  { k: '본전 실패 가드 (무장 손익 · 반납 · 확인)', v: { conservative: '+0.6% · 0.45% · 20초', neutral: '+1.0% · 0.8% · 60초', aggressive: '+1.2% · 1.0% · 45초' }, hint: 'BREAKEVEN_FAIL_ARM_PNL / GIVEBACK_PCT / CONFIRM_SECONDS' },
  { k: '무추세 청산 (보유 · 고점 미만 · 손익 이하 · 확인)', v: { conservative: '15분 · +0.2% · -0.35% · 60초', neutral: '20분 · +0.3% · -0.45% · 90초', aggressive: '25분 · +0.5% · -0.8% · 120초' }, hint: 'NO_TREND_EXIT_*' },
];

/** "!" next to the profile select: a toggle that shows what each profile changes */
/** children: what stays right beside the "!" (dry-run, user 2026-11-25) - the description panel opens below them on its own line */
function ProfileInfo({ current, children }: { current: RiskProfile; children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className={`mini info${open ? ' on' : ''}`} title="프로파일 3개가 무엇을 바꾸는지 보기" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        !
      </button>
      {children}
      {open && (
        <div className="profile-info">
          <ul className="small">
            {RISK_PROFILES.map((p) => (
              <li key={p} className={p === current ? 'cur' : ''}>
                <b>{p}</b> {PROFILE_SUMMARY[p]}
                {p === current && <span className="pill live">선택됨</span>}
              </li>
            ))}
          </ul>
          <table className="grid compact">
            <thead>
              <tr>
                <th>항목</th>
                {RISK_PROFILES.map((p) => (
                  <th key={p} className={p === current ? 'cur' : ''}>
                    {p}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PROFILE_ROWS.map((r) => (
                <tr key={r.k} className="nohover" title={r.hint || ''}>
                  <td>{r.k}</td>
                  {RISK_PROFILES.map((p) => (
                    <td key={p} className={p === current ? 'cur' : ''}>
                      {r.v[p]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            모든 프로파일에서 같은 것: 하드스탑 -3.5% (매수 4분 뒤 활성, 10초 확인), ATR 손절 1.5×ATR·최소 -3.5% (4분 뒤, 20초 확인), 1차 익절 +3.0%에 40% (ATR 연동), 이익 잠금 +1.0% 뒤
            -0.4% 반납 시 전량, 15:20 정규장 청산 (자동매매 시간이 08:00~20:00 이면 NXT 가능 종목은 19:59:50). 값은 서버 live_trading/risk_profiles/*.json 기준이며 바꾼 프로파일은 다음 실행/재시작부터
            적용됩니다.
          </p>
        </div>
      )}
    </>
  );
}

const HOST_NAME: Record<RunMode, string> = { local: '이 PC', remote: '라즈베리파이' };

/**
 * 매매 실행 위치: both hosts as radio rows (role, r003, tick collector). The checked row is the host the [실행] button starts r003
 * on (= settings.runMode / the open connection); checking the other row saves runMode and reconnects (what 설정/연결 > 실행 위치 >
 * [저장 후 연결] did). Role switches, collector start, the other host's r003 stop and the watchlist copy stay under "고급".
 */
function HostPanel() {
  const control = useStore((s) => s.control);
  const status = useStore((s) => s.status);
  const settings = useStore((s) => s.settings);
  const bumpHistory = useStore((s) => s.bumpHistory);
  const connected = status.conn === 'connected';
  const mode: RunMode = (connected && status.mode) || settings?.runMode || 'remote';
  const peerKind: RunMode = mode === 'local' ? 'remote' : 'local';
  const self = connected ? control.status : null;
  // [2026-11-18] last answers (dataCache) shown at once when the tab is opened again, then refreshed
  const [peer, setPeer] = useState<R016Status | null>(() => peek<R016Status>('control:peer') || null);
  const [peerErr, setPeerErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [msg, setMsg] = useState('');
  const ctlBusy = !!control.busy;

  const loadPeer = async () => {
    setBusy(true);
    setPeerErr('');
    const r = await window.api.peerStatus();
    setBusy(false);
    if (r.ok && r.data) setPeer(put('control:peer', r.data));
    else {
      setPeer(null);
      setPeerErr(r.notConfigured ? `설정되지 않음 (${r.error})` : `상태 확인 실패: ${r.error}`);
    }
  };
  useEffect(() => {
    if (connected) void loadPeer();
    else {
      setPeer(null);
      setPeerErr('');
    }
  }, [connected, mode]); // eslint-disable-line react-hooks/exhaustive-deps

  /** make `kind` the host that [실행] starts r003 on: save runMode, then connect (the view, the control and the scheduler follow) */
  const switchTo = async (kind: RunMode) => {
    if (!settings || switching) return;
    setSwitching(true);
    setMsg(kind === mode ? `${HOST_NAME[kind]}에 연결 중…` : `실행 위치를 ${HOST_NAME[kind]}로 바꾸는 중…`);
    const s = await window.api.saveSettings({ ...settings, runMode: kind });
    if (!s.ok) {
      setMsg(`실패: ${s.error}`);
      setSwitching(false);
      return;
    }
    useStore.setState({ settings: s.settings });
    const r = await window.api.connect();
    setSwitching(false);
    setMsg(r.ok ? `이제 [실행]은 ${HOST_NAME[kind]}에서 r003을 시작합니다.` : `${HOST_NAME[kind]} 연결 실패: ${r.error} - 설정/연결 탭에서 ${HOST_NAME[kind]} 연결 정보를 확인하세요.`);
  };

  const act = async (fn: () => Promise<{ ok: boolean; error?: string; cancelled?: boolean }>, okMsg: string, reloadPeer = true) => {
    setBusy(true);
    setMsg('');
    const r = await fn();
    setBusy(false);
    if (r.cancelled) return;
    setMsg(r.ok ? okMsg : `실패: ${r.error}`);
    bumpHistory();
    if (reloadPeer) await loadPeer();
  };
  const restartCollector = async () => {
    setMsg('틱 수집기 재시작 중…');
    const res = await window.api.ctlCollectorRestart();
    const d = res.data || {};
    setMsg(res.ok ? (d.restarted ? `틱 수집기를 현재 감시목록으로 다시 시작했습니다 (pid ${d.pid})` : `틱 수집기를 멈췄습니다${d.window ? ` - 다시 시작 안 함: ${d.window}` : ''}`) : `실패: ${res.error}`);
  };
  const startCollector = async () => {
    setMsg('틱 수집기 시작 중…');
    const res = await window.api.ctlCollectorStart(false);
    const d = res.data as { already_running?: boolean; pid?: number } | undefined;
    setMsg(res.ok ? (d?.already_running ? `이미 실행 중 (pid ${d.pid})` : `시작했습니다 (pid ${d?.pid})`) : `실패: ${res.error}`);
  };

  const selfRole = self?.host?.role?.role;
  const peerRole = peer?.host?.role?.role;
  const peerName = HOST_NAME[peerKind];
  const anyBusy = busy || switching || ctlBusy;

  const row = (kind: RunMode) => {
    const selected = kind === mode;
    const st = selected ? self : peer;
    const role = st?.host?.role?.role;
    const s = botSummary(st);
    const hostText = st?.host ? st.host.hostname : kind === 'local' ? 'Windows' : 'Linux · SSH';
    const coll = st?.collector;
    return (
      <label key={kind} className={`host-row${selected ? ' sel' : ''}`} title={selected ? '[실행]을 누르면 이 호스트에서 r003이 시작됩니다' : `실행 위치를 ${HOST_NAME[kind]}로 바꿉니다 (설정 저장 후 다시 연결)`}>
        <input type="radio" name="run-host" checked={selected} disabled={anyBusy} onChange={() => void switchTo(kind)} />
        <span className="hn">
          {HOST_NAME[kind]}
          <span>{hostText}</span>
        </span>
        <span className={role === 'standby' ? 'muted' : ''} title={st?.host?.role?.set ? `${st.host.role.by || ''} ${st.host.role.at || ''} ${st.host.role.note || ''}` : st?.host ? '역할 파일 없음 = primary' : ''}>
          {st?.host ? (role === 'standby' ? 'standby' : 'primary') : st ? '이전 버전 (git pull 필요)' : '-'}
        </span>
        <span>
          {st ? (
            <span className={`bot bot-${s}`}>
              <i className="dot" />
              {BOT_LABEL[s]}
            </span>
          ) : (
            <span className="muted small">{selected ? (connected ? '상태 확인 중…' : '연결 안 됨') : peerErr || '확인 안 됨'}</span>
          )}
        </span>
        <span className="muted small">
          {st && (
            <>
              틱 수집기 {coll?.running ? `실행 중${coll.symbols ? ` (${coll.symbols}종목)` : ''}` : '정지'}
              {coll?.running && !!coll.missing?.length && (
                <>
                  {' '}
                  <span className="warn-text" title={`수집기가 시작된 뒤 감시목록에 들어온 종목 - 틱 파일이 없어 1분봉 트리거가 평가되지 않습니다: ${coll.missing.join(', ')}`}>
                    구독 누락 {coll.missing.length}종목
                  </span>{' '}
                  {selected && (
                    <button className="mini" disabled={anyBusy || !connected} onClick={(e) => (e.preventDefault(), void restartCollector())}>
                      수집기 재시작
                    </button>
                  )}
                </>
              )}
            </>
          )}
        </span>
      </label>
    );
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <span className="title">매매 실행 위치</span>
          <span className="sub">r003은 계좌당 한 호스트에서만 실행됩니다</span>
        </div>
        <div className="card-actions">
          {!connected && (
            <button className="primary" disabled={switching} onClick={() => void switchTo(mode)}>
              {HOST_NAME[mode]}에 연결
            </button>
          )}
          <button disabled={anyBusy || !connected} onClick={() => void loadPeer()}>
            {peerName} 상태 확인
          </button>
        </div>
      </div>
      <div className="host-rows">
        {row('local')}
        {row('remote')}
      </div>
      {(msg || busy || switching) && (
        <div className="card-row">
          {(busy || switching) && <i className="spinner" />}
          {msg && <span className={/실패/.test(msg) ? 'err-text small' : 'muted small'}>{msg}</span>}
        </div>
      )}
      <details className="adv card-adv">
        <summary>고급: 역할 바꾸기 · 틱 수집기 · {peerName} r003 정지 · 감시목록 복사</summary>
        <p className="muted small">
          [실행]은 시작 전에 {peerName}를 확인해 r003이 돌고 있으면 거부하고, primary 이면 standby 로 바꾼 뒤 {HOST_NAME[mode]}를 primary 로 해서 시작합니다 (standby 호스트는
          r003 시작을 거부하고 틱 수집기 cron 을 띄우지 않습니다). 아래 버튼은 그 역할을 손으로 바꿀 때만 씁니다.
        </p>
        <div className="actions">
          <span className="muted small">{HOST_NAME[mode]}:</span>
          {selfRole === 'standby' ? (
            <button className="mini" disabled={anyBusy || !connected} onClick={() => void act(() => window.api.ctlRole('primary'), `${HOST_NAME[mode]}를 primary 로 바꿨습니다.`, false)}>
              primary 로
            </button>
          ) : (
            <button
              className="mini"
              disabled={anyBusy || !connected || !self?.host || !!self?.running}
              title={self?.running ? 'r003 실행 중에는 standby 로 바꿀 수 없습니다' : ''}
              onClick={() => void act(() => window.api.ctlRole('standby'), `${HOST_NAME[mode]}를 standby 로 바꿨습니다.`, false)}
            >
              standby 로
            </button>
          )}
          {mode === 'local' && !self?.collector?.running && (
            <button className="mini" disabled={anyBusy || !connected || selfRole === 'standby'} onClick={() => void startCollector()}>
              틱 수집기 시작
            </button>
          )}
          <span className="muted small">{peerName}:</span>
          <button className="mini" disabled={anyBusy || !peer?.running} onClick={() => void act(() => window.api.peerStop(), `${peerName} r003 을 정지했습니다.`)}>
            r003 정지
          </button>
          <button className="mini" disabled={anyBusy || !peer?.host || peerRole === 'standby' || !!peer?.running} onClick={() => void act(() => window.api.peerRole('standby'), `${peerName}를 standby 로 바꿨습니다.`)}>
            standby 로
          </button>
          <button className="mini" disabled={anyBusy || !peer?.host || peerRole !== 'standby'} onClick={() => void act(() => window.api.peerRole('primary'), `${peerName}를 primary 로 바꿨습니다.`)}>
            primary 로
          </button>
          <button className="mini" disabled={anyBusy || !peer} onClick={() => void act(() => window.api.peerSyncWatchlist(), `감시목록을 ${peerName}로 보냈습니다.`)}>
            감시목록 보내기 →
          </button>
        </div>
        <p className="muted small">
          노트북을 못 쓰는 날: 라즈베리파이 줄을 고른 뒤 [라즈베리파이에서 실행]. 노트북이 켜져 있으면 자동으로 standby 가 됩니다. 장 마감 후 자동 작업이 끝나면 새 감시목록이
          라즈베리파이로도 복사됩니다 (설정).
        </p>
      </details>
    </section>
  );
}

export function ControlView() {
  const control = useStore((s) => s.control);
  const connected = useStore((s) => s.status.conn === 'connected');
  const isLocal = useStore((s) => s.status.mode === 'local');
  const historyRev = useStore((s) => s.historyRev);
  const bumpHistory = useStore((s) => s.bumpHistory);
  const st = control.status;
  const [mode, setMode] = useState<{ dryRun: boolean; profile: RiskProfile } | null>(null);
  const [history, setHistory] = useState<HistoryRow[]>(() => peek<HistoryRow[]>('control:history') || []);
  const [histErr, setHistErr] = useState('');
  const [overrides, setOverrides] = useState<SchemaKey[]>(() => peek<SchemaKey[]>('control:overrides') || []);
  const [msg, setMsg] = useState('');

  // form follows the server until the user changes it
  const dryRun = mode?.dryRun ?? st?.dry_run ?? false;
  const paper = control.paper;
  const strategy = control.strategy;
  const tradeMode = tradeModeOf(strategy?.effective, paper?.effective);
  const runningMode = tradeModeOf(strategy?.running ?? strategy?.effective, paper?.running ?? paper?.effective);
  const modePending = !!strategy && strategy.running !== null && (strategy.pending || !!paper?.pending || runningMode !== tradeMode);
  const [modeBusy, setModeBusy] = useState(false);
  const setTradeMode = async (next: TradeMode) => {
    const changes = tradeModeChanges(next);
    if (!changes || next === tradeMode) return;
    if (
      next === 'edge' &&
      !window.confirm(
        'Edge AI 방식은 모델 신호만으로 실제 주문을 냅니다.\n\n' +
          '과거 검증(2026-08-14~11-11, 58일)에서는 모든 확률 기준에서 거래당 -0.43 ~ -0.58% 손실이었습니다.\n' +
          '매도는 +2.0% 익절 / -1.7% 손절 / 60분 경과로 고정됩니다. 일일 손실 한도와 동시 보유 상한은 그대로 적용됩니다.\n\n' +
          'r003 재시작 후부터 적용됩니다. 바꿀까요?',
      )
    )
      return;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet(changes);
      if (!r.ok) window.alert(`매매 방식을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setModeBusy(false);
    }
  };
  // [2026-11-25] 매매 방식 = any combination of 규칙 매매 (server RULE_BUY_ENABLED) / Edge AI 가상 매매 (EDGE_ONLY_MODE paper) /
  // 반등 알림 (AutoBotHtsH app setting; AutoBotH has no rebound watcher). TRADING_STRATEGY_MODE is not touched by the checkboxes
  const ruleBuy = control.ruleBuy;
  const normalMode = !strategy || tradeMode === 'rules' || tradeMode === 'rules+paper'; // not read yet: the (disabled) checkboxes
  const rulesOn = ruleBuy ? ruleBuy.effective === 'true' : true;
  const rulesRunning = ruleBuy ? (ruleBuy.running ?? ruleBuy.effective) === 'true' : true;
  const rulesPending = !!ruleBuy && ruleBuy.running !== null && (ruleBuy.pending || rulesRunning !== rulesOn);
  const paperOn = tradeMode === 'rules+paper';
  const paperRunning = runningMode === 'rules+paper';
  const rb = reboundApi();
  const [reboundOn, setReboundOn] = useState<boolean | null>(null);
  useEffect(() => {
    if (!rb) return;
    let alive = true;
    const load = () =>
      void rb
        .reboundStatus()
        .then((s) => alive && setReboundOn(!!s.enabled))
        .catch(() => undefined);
    load();
    const id = window.setInterval(load, 10000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const kinds: TradeKind[] = ['rules', 'paper']; // [2026-11-25] 매매 방식 = these two; 반등 알림 is its own group
  const comboNow = { rules: rulesOn, paper: paperOn, rebound: !!reboundOn };
  const comboRunning = { rules: rulesRunning, paper: paperRunning, rebound: !!reboundOn };
  const setKind = async (kind: TradeKind, on: boolean) => {
    if (kind === 'rebound') {
      if (!rb) return;
      const s = await rb.reboundEnable(on);
      setReboundOn(!!s.enabled);
      return;
    }
    const next = { rules: kind === 'rules' ? on : rulesOn, paper: kind === 'paper' ? on : paperOn };
    const changes = comboChanges(next, { rules: rulesOn, paper: paperOn }, !!ruleBuy);
    if (!changes) return;
    // [2026-11-25] Edge AI 매매 off: its live orders go off with it (stored too - nothing stays on out of sight)
    // ... and on: it starts with the live orders off too - they are switched on by their own checkbox and its warning only
    if (kind === 'paper' && control.edgeLive?.effective === 'true') changes[EDGE_LIVE_KEY] = false;
    const sellMode = autoSellOf(control.autoSell?.effective);
    const text =
      kind === 'rules'
        ? on
          ? '규칙 매매를 켭니다. 규칙 매수 조건을 통과하면 실제 매수 주문을 냅니다.'
          : '규칙 매매를 끕니다. 규칙 매수 조건을 통과해도 신규 매수 주문을 내지 않습니다(로그만).\n' +
            `이미 보유한 종목의 매도는 자동 매도 설정(${AUTO_SELL_KO[sellMode]})대로 계속됩니다.` +
            (sellMode === 'all' ? '' : '\n자동 매도가 전체가 아니므로 보유 종목이 남을 수 있습니다 - 직접 매도해야 합니다.')
        : on
          ? 'Edge AI 신호로 가상 매매를 합니다 (실제 주문 없음, 가상 매매 탭).' +
            (control.edgeLive?.effective === 'true' ? '\n저장돼 있던 Edge AI 실제 매매 설정은 끈 상태로 시작합니다 (켜려면 "실제 매매" 체크).' : '')
          : '신규 가상 매수를 멈춥니다. 오늘 열린 가상 보유는 마감 청산까지 그대로 청산됩니다.' +
            (control.edgeLive?.effective === 'true'
              ? '\nEdge AI 실제 매매도 함께 끕니다 (이미 산 Edge 수량의 청산은 계속됩니다).'
              : '');
    const restart = 'r003 재시작 후부터 적용됩니다. 미체결 주문이 있으면 정리된 뒤(또는 장 마감 후) 재시작하세요 - 재시작하면 미체결 주문 추적이 끊깁니다.';
    if (!window.confirm(`${TRADE_KIND_KO[kind]}: ${on ? '켬' : '끔'}\n\n${text}\n\n${restart}\n\n바꿀까요?`)) return;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet(changes);
      if (!r.ok) window.alert(`매매 방식을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setModeBusy(false);
    }
  };
  /** the server runs another strategy (edge = Edge AI real orders, basic): back to the rule-based strategy */
  const toNormal = async () => {
    if (
      !window.confirm(
        `서버 매매 방식(${strategy?.effective || '-'})을 규칙 매매 방식(normal)으로 바꿉니다.\n\n` +
          '이미 보유한 종목의 매도 규칙도 규칙 매매 방식으로 바뀝니다. Edge AI 가상 매매는 꺼진 상태로 시작합니다.\n\nr003 재시작 후부터 적용됩니다. 바꿀까요?',
      )
    )
      return;
    await setTradeMode('rules');
  };
  // [2026-11-12] the Edge AI model in use (current_meta.json of the connected host), re-read every minute like the bot does
  const [edgeModel, setEdgeModel] = useState<EdgeModelInfo | null | undefined>(() => peek<EdgeModelInfo | null>('control:edge'));
  useEffect(() => {
    if (!connected) return setEdgeModel(undefined);
    let alive = true;
    const load = () =>
      void window.api.getEdgeModel().then((r) => {
        if (r.ok) put('control:edge', r.model);
        if (alive) setEdgeModel(r.ok ? r.model : peek<EdgeModelInfo | null>('control:edge'));
      });
    load();
    const id = window.setInterval(load, 60000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [connected, isLocal]);
  // [2026-11-12] AUTO_SELL_MODE: which sells the bot makes by itself (all / only the net -2% stop / none)
  const autoSell = control.autoSell;
  const autoSellMode = autoSellOf(autoSell?.effective);
  const autoSellRunning = autoSellOf(autoSell?.running ?? autoSell?.effective);
  const autoSellPending = !!autoSell && autoSell.running !== null && (autoSell.pending || autoSellRunning !== autoSellMode);
  const setAutoSell = async (next: AutoSellMode) => {
    if (next === autoSellMode) return;
    // [2026-11-25] Edge AI live orders need the full automatic sells (the server refuses it too)
    if (next !== 'all' && control.edgeLive?.effective === 'true') {
      window.alert('Edge AI 실제 매매가 켜져 있어 자동 매도를 바꿀 수 없습니다.\n먼저 Edge AI 매매의 "실제 매매"를 끈 뒤 바꾸세요.');
      return;
    }
    const text =
      next === 'all'
        ? '봇이 익절·트레일링·손절·당일 청산(15:20 / NXT 19:59)을 모두 자동으로 합니다.\n서버 EXIT_ENGINE_MODE=live(기본)이면 적응형 청산 엔진(분할 익절·고점 되돌림·손절·15:15 청산)이 매도를 판정합니다.'
        : next === 'stop_only'
          ? '봇은 수수료+거래세 포함 순손익 -2% 손절만 자동으로 매도합니다.\n익절·트레일링·당일 청산은 하지 않으므로 직접 매도해야 하고, 팔지 않으면 다음 날로 넘어갑니다.'
          : '봇은 어떤 매도도 하지 않습니다(손절 포함). 모든 매도를 직접 해야 하고, 팔지 않으면 다음 날로 넘어갑니다.';
    const edgeNote = control.edgeLive ? `\n\n단, ${EDGE_LIVE_EXIT_NOTE}` : '';
    if (!window.confirm(`자동 매도: ${AUTO_SELL_KO[next]}\n\n${text}${edgeNote}\n\nr003 재시작 후부터 적용됩니다. 바꿀까요?`)) return;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet({ [AUTO_SELL_KEY]: next });
      if (!r.ok) window.alert(`자동 매도 설정을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setModeBusy(false);
    }
  };
  // [2026-11-23] EDGE_ONLY_LOG_ONLY (user 2026-11-23: paper trading that is on should paper-buy): the checkbox beside 매매 방식
  // switches the new paper buys on / off; off = candidates are recorded only (the server default since 2026-11-20)
  const logOnly = control.paperLogOnly;
  const paperBuys = logOnly ? logOnly.effective !== 'true' : false;
  const paperBuysRunning = logOnly ? (logOnly.running ?? logOnly.effective) !== 'true' : false;
  const paperBuysPending = !!logOnly && logOnly.running !== null && (logOnly.pending || paperBuysRunning !== paperBuys);
  const setPaperBuys = async (on: boolean) => {
    const text = on
      ? 'Edge AI 신호가 뜨면 가상 매수 / 가상 청산을 기록합니다 (실제 주문 없음).'
      : '신규 가상 매수를 하지 않고, 신호가 뜬 봉을 후보로만 기록합니다. 이미 있는 가상 보유는 그대로 청산됩니다.';
    if (!window.confirm(`가상 매수 실행: ${on ? '켬' : '끔'}\n\n${text}\n\nr003 재시작 후부터 적용됩니다. 바꿀까요?`)) return;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet({ [PAPER_LOG_ONLY_KEY]: !on });
      if (!r.ok) window.alert(`가상 매수 설정을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setModeBusy(false);
    }
  };
  // [2026-11-25] EDGE_LIVE_ORDERS: "가상 매수" unchecked = Edge AI live orders beside the rule strategy (real orders, small limits).
  // A server without the key keeps the checkbox locked on 가상 매수
  const edgeLive = control.edgeLive;
  const liveOn = edgeLive ? edgeLive.effective === 'true' : false;
  const liveRunning = edgeLive?.running === 'true'; // the running r003's value only (stopped / unknown: not running)
  const livePending = !!edgeLive && edgeLive.running !== null && (edgeLive.pending || liveRunning !== liveOn);
  const liveLimits = control.edgeLiveLimits;
  const liveSummary = parseEdgeLiveSummary(st);
  const liveHeld = !!liveSummary && (liveSummary.held.length > 0 || liveSummary.openBuys.length > 0);
  const setEdgeLive = async (on: boolean) => {
    if (on && autoSellMode !== 'all') {
      window.alert('Edge AI 실제 매매는 자동 매도가 "전체 자동"일 때만 켤 수 있습니다 (신규 Edge 매수의 조건).');
      return;
    }
    const lim = liveLimits;
    const limText = lim
      ? `1회 ${lim.order.effective.toLocaleString('ko-KR')}원 · 동시 ${lim.positions.effective}종목 · 하루 ${lim.entries.effective}회 · 하루 손실 ${lim.loss.effective.toLocaleString('ko-KR')}원에서 중단`
      : '서버 기본 한도';
    const text = on
      ? `${EDGE_LIVE_ON_WARNING.replace(/\n\n켤까요\?$/, '')}\n\n` +
        `한도: ${limText}\n` +
        '주문: 정규장 14:50 까지, 매도1호가 지정가 1회 (재주문 없음). 청산: 아래 청산 기준 / 15:20.\n' +
        'Edge 가 산 수량은 Edge 규칙으로만 팔고, 같은 종목의 직접 매수분은 팔지 않습니다.\n' +
        '켜져 있는 동안 가상 매매는 후보만 기록합니다.'
      : 'Edge AI 실제 매매를 끕니다 (가상 매매는 아래 체크대로).\n이미 Edge 가 산 수량이 있으면 그 청산(손절 / 익절 / 시간 / 15:20)은 계속됩니다.';
    const restart = 'r003 재시작 후부터 적용됩니다. 미체결 주문이 있으면 정리된 뒤(또는 장 마감 후) 재시작하세요.';
    if (!window.confirm(`Edge AI 실제 매매: ${on ? '켬' : '끔'}\n\n${text}\n\n${restart}\n\n바꿀까요?`)) return;
    if (on && !window.confirm('한 번 더 확인합니다.\n\n실제 계좌로 Edge AI 매수 주문이 나갑니다. 켤까요?')) return;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet({ [EDGE_LIVE_KEY]: on });
      if (!r.ok) window.alert(`Edge AI 실제 주문 설정을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setModeBusy(false);
    }
  };
  // [2026-11-13] AUTO_SELL_SCOPE: the automatic sells sell only what the bot bought / the whole holding (hand-bought included)
  const sellScope = control.autoSellScope;
  const sellScopeMode = autoSellScopeOf(sellScope?.effective);
  const sellScopeRunning = autoSellScopeOf(sellScope?.running ?? sellScope?.effective);
  const sellScopePending = !!sellScope && sellScope.running !== null && (sellScope.pending || sellScopeRunning !== sellScopeMode);
  const setSellScope = async (next: AutoSellScope) => {
    if (next === sellScopeMode) return;
    const text =
      next === 'all'
        ? '봇이 관리하는 종목(오늘 감시목록에 있고 오늘 매수된 종목)은 계좌 수량 전부를 자동 매도합니다.\nMTS 등으로 직접 매수한 수량도 함께 팔립니다.'
        : '봇이 매수한 수량만 자동 매도합니다.\n같은 종목을 MTS 등으로 더 샀으면 봇 수량만 팔고 직접 매수분은 남기며, 전부 직접 매수한 종목은 매도 평가를 하지 않습니다.';
    if (!window.confirm(`자동 매도 대상: ${AUTO_SELL_SCOPE_KO[next]}\n\n${text}\n\nr003 재시작 후부터 적용됩니다. 바꿀까요?`)) return;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet({ [AUTO_SELL_SCOPE_KEY]: next });
      if (!r.ok) window.alert(`자동 매도 대상을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setModeBusy(false);
    }
  };
  const profile = mode?.profile ?? ((st?.risk_profile as RiskProfile) || 'neutral');
  const busy = !!control.busy;
  const btn = serviceButtons(st, busy || !connected);
  const summary = botSummary(st);
  const svc = st?.service;
  const nPending = pendingCount(st);
  const hostName = isLocal ? '이 PC' : '라즈베리파이';

  useEffect(() => {
    if (!connected) return;
    void window.api.ctlStatus();
  }, [connected]);

  useEffect(() => {
    if (!connected) return;
    let alive = true;
    void window.api.ctlHistory(20).then((r) => {
      if (!alive) return;
      if (r.ok && r.data) {
        setHistory(put('control:history', [...r.data.rows].reverse()));
        setHistErr('');
      } else setHistErr(r.error || '이력 조회 실패');
    });
    void window.api.ctlSchema(false).then((r) => {
      if (r.ok && r.data) put('control:overrides', r.data.keys.filter((k) => k.has_override));
      if (alive && r.ok && r.data) setOverrides(r.data.keys.filter((k) => k.has_override));
    });
    return () => {
      alive = false;
    };
  }, [connected, historyRev, st?.override_count, st?.pending_restart, st?.service?.started_at]);

  const run = async (verb: 'start' | 'stop' | 'restart') => {
    setMsg('');
    const r = await window.api.ctlService(verb, verb === 'stop' ? {} : { dryRun, profile });
    if (r.cancelled) return;
    setMsg(r.ok ? '' : `실패: ${r.error}`);
    if (r.ok) setMode(null);
    bumpHistory();
  };

  const unset = async (key: string) => {
    const r = await window.api.ctlUnset([key]);
    setMsg(r.ok ? `${key} 기본값으로 되돌림 - 재시작 후 반영` : `실패: ${r.error}`);
    bumpHistory();
  };

  const reset = async () => {
    const r = await window.api.ctlReset();
    if (r.cancelled) return;
    setMsg(r.ok ? '모든 오버라이드를 지웠습니다 - 재시작 후 반영' : `실패: ${r.error}`);
    bumpHistory();
  };

  const pendingRows = useMemo(() => {
    const rows: { k: string; text: string; key?: string; pending?: boolean }[] = [];
    for (const o of overrides) {
      const h = pctHint(o.key, o.override);
      rows.push({
        k: o.key,
        key: o.key,
        pending: o.pending,
        text: `${o.key} = ${fmtValue(o.override)}${h ? ` (${h.slice(2)})` : ''}  (기본 ${fmtValue(o.base)})`,
      });
    }
    if (st?.pending?.env?.changed_since_start) rows.push({ k: 'env', pending: true, text: `실행 환경(r003.env) 변경 ${isoShow(st.pending.env.mtime)}` });
    if (st?.pending?.watchlist?.changed_since_start) rows.push({ k: 'wl', pending: true, text: `감시목록 변경 ${isoShow(st.pending.watchlist.mtime)}` });
    return rows;
  }, [overrides, st]);

  if (!connected && !st) {
    return (
      <div className="control cards">
        <HostPanel />
        <div className="empty">연결되어 있지 않습니다 - 위에서 실행 위치를 고르고 [연결]을 누르세요 (연결 정보는 설정/연결 탭).</div>
      </div>
    );
  }

  // mode cell: dry_run is null while the bot is stopped (no running snapshot) - never show that as 실주문
  const modeCell =
    st?.dry_run === true ? (
      <span>dry-run</span>
    ) : st?.dry_run === false ? (
      <span className="live-mode">실주문</span>
    ) : (
      <span className="muted" title={st?.running ? '실행 스냅샷이 없어 실행 모드를 확인할 수 없습니다 (재시작 후 정확)' : '정지 중 - 다음 실행 모드는 아래 dry-run 체크'}>
        {st?.running ? '확인 불가' : '-'}
      </span>
    );
  const noteRight = [
    st?.external_r003?.length ? `외부 r003 ${st.external_r003.length}개 감지` : '외부 r003 없음',
    isLocal ? '크래시 시 60초 후 재시작 (10분 3회까지)' : '서버 재부팅 후에는 여기서 [실행]',
  ].join(' · ');

  return (
    <div className="control cards">
      <HostPanel />

      <section className="card">
        <div className="card-head">
          <div>
            <span className="title">봇 상태</span>
            <span className="sub">r003 · {svc?.unit || 'autobot-r003.service'}</span>
          </div>
          <div className="card-actions">
            {busy && (
              <span className="busy small">
                <i className="spinner" />
                {BUSY_LABEL[control.busy!] || `${control.busy} 진행 중…`}
              </span>
            )}
            <button disabled={!btn.restart} onClick={() => void run('restart')}>
              재시작
            </button>
            <button className={st?.running ? 'danger-text' : ''} disabled={!btn.stop} onClick={() => void run('stop')}>
              정지
            </button>
            <button className="primary" disabled={!btn.start} title={btn.startReason || ''} onClick={() => void run('start')}>
              {hostName}에서 실행
            </button>
          </div>
        </div>
        <div className="stats">
          <div>
            <span className="k">상태</span>
            <span className={`v bot bot-${summary}`}>
              <i className="dot" />
              {BOT_LABEL[summary]}
            </span>
          </div>
          <div>
            <span className="k">모드</span>
            <span className="v">{modeCell}</span>
          </div>
          <div>
            <span className="k">시작</span>
            <span className="v" title={isoShow(svc?.started_at)}>
              {timeOnly(svc?.started_at)}
            </span>
          </div>
          <div>
            <span className="k">PID · 재시작</span>
            <span className="v">
              {svc?.main_pid ?? '-'} · {svc?.restarts ?? 0}회
            </span>
          </div>
        </div>
        <div className="card-row">
          <label>
            프로파일{' '}
            <select value={profile} disabled={busy} onChange={(e) => setMode({ dryRun, profile: e.target.value as RiskProfile })}>
              {RISK_PROFILES.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <ProfileInfo current={profile}>
            {/* [2026-11-25] dry-run right beside the "!" whether the description is open or not (user 2026-11-25) */}
            <label className="check">
              <input type="checkbox" checked={dryRun} disabled={busy} onChange={(e) => setMode({ dryRun: e.target.checked, profile })} /> dry-run
            </label>
            {mode && (mode.dryRun !== st?.dry_run || mode.profile !== st?.risk_profile) && <span className="muted small">다음 실행/재시작부터 적용 (r003.env에 저장)</span>}
          </ProfileInfo>
        </div>
        {/* [2026-11-25] 매매 방식 (user 2026-11-25): one block per feature - 규칙 매매 / Edge AI 매매 / 반등 알림 - with its own options
            indented under it; the sell settings are their own group */}
        <div className="trade-box">
          {/* [2026-11-25] user: the exits stand above the trading kinds (not under Edge AI); 반등 알림 is its own group */}
          {control.paperExit && (
            <>
              <div className="tk-head">
                <b>청산 기준</b>
                <span className="muted small">Edge AI 매매(실제·가상)의 익절 / 손절 / 보유 시간 · 규칙 매매는 아래 매도 설정의 자동 매도 규칙으로 청산</span>
              </div>
              <div className="tk">
                <PaperExitEdit st={control.paperExit} model={edgeModel || null} real={liveOn || liveRunning || liveHeld || tradeMode === 'edge'} disabled={busy || !connected || modeBusy} />
              </div>
            </>
          )}
          <div className={control.paperExit ? 'tk-head tk-gap' : 'tk-head'}>
            <b>매매 방식</b>
            {strategy && normalMode && (
              <span className={comboLabel(comboNow, kinds) === '모두 꺼짐' ? 'badge-warn' : 'muted small'}>
                {rulesPending || modePending
                  ? `→ r003 재시작 후 적용 (실행 중: ${
                      strategy.running && strategy.running !== 'normal' ? `서버 설정 ${strategy.running}` : comboLabel(comboRunning, kinds)
                    })`
                  : comboLabel(comboNow, kinds) === '모두 꺼짐'
                    ? '모두 꺼짐 - 신규 매수 없음 (보유 매도는 자동 매도 설정대로)'
                    : ''}
              </span>
            )}
            {!normalMode && (
              <>
                <span className="badge-warn">
                  서버 설정: {tradeMode === 'edge' ? 'Edge AI 실주문 (edge)' : strategy?.effective || '-'}
                  {modePending && ` → r003 재시작 후 적용 (실행 중: ${strategy?.running || '-'})`}
                </span>
                <button className="mini" disabled={busy || !connected || modeBusy} onClick={() => void toNormal()}>
                  규칙 매매 방식으로 전환
                </button>
              </>
            )}
          </div>
          {normalMode && (
            <div className="tk">
              <label className="check tk-main" title={ruleBuy ? '규칙 매수 조건(r005)으로 실제 매수 주문 (서버 RULE_BUY_ENABLED)' : '서버 코드 갱신 후 끌 수 있습니다'}>
                <input type="checkbox" checked={rulesOn} disabled={busy || !connected || modeBusy || !ruleBuy} onChange={(e) => void setKind('rules', e.target.checked)} />{' '}
                {TRADE_KIND_KO.rules}
              </label>
              <span className="muted small">규칙 매수 조건으로 실제 매수 주문 · r003 재시작 후 적용</span>
            </div>
          )}
          {normalMode && (
            <div className="tk">
              <label className="check tk-main" title="Edge AI 모델 신호로 매수 / 청산 (서버 EDGE_ONLY_MODE)">
                <input type="checkbox" checked={paperOn} disabled={busy || !connected || modeBusy || !paper} onChange={(e) => void setKind('paper', e.target.checked)} />{' '}
                Edge AI 매매
              </label>
              <span className="muted small">{paper ? '모델 신호로 매수 / 청산 · r003 재시작 후 적용' : '서버 코드 갱신 후 선택할 수 있습니다'}</span>
              {!paperOn && (liveHeld || liveRunning) && (
                <div className="tk-sub">
                  <div className="tk-opt">
                    <span className="badge-warn" title="Edge AI 매매는 꺼져 있지만 Edge 가 산 수량은 Edge 규칙으로 청산됩니다">
                      {liveRunning ? '실행 중인 r003 은 재시작 전까지 Edge AI 실제 매매 설정으로 동작합니다. ' : ''}
                      {liveSummary ? `실제 매매 현황: ${edgeLiveSummaryText(liveSummary, null)}` : '실제 매매 현황을 읽을 수 없습니다 (체결 장부 없음)'}
                    </span>
                  </div>
                </div>
              )}
              {paperOn && (
                <div className="tk-sub">
                  <div className="tk-opt">
                    <label
                      className="check"
                      title={
                        edgeLive
                          ? 'Edge AI 신호로 실제 주문 (서버 EDGE_LIVE_ORDERS, 규칙 매매와 병행, 작은 한도). r003 재시작 후 적용'
                          : '실제 매매는 서버 코드 갱신 후 쓸 수 있습니다'
                      }
                    >
                      <input type="checkbox" checked={liveOn} disabled={busy || !connected || modeBusy || !edgeLive} onChange={(e) => void setEdgeLive(e.target.checked)} /> 실제 매매
                    </label>
                    {liveOn ? (
                      <span className="badge-warn">Edge AI 신호로 실제 주문 (규칙 매매와 병행)</span>
                    ) : (
                      <span className="muted small">{edgeLive ? '실제 계좌로 주문 · 작은 한도 · 자동 매도가 전체일 때만' : '서버 코드 갱신 후 쓸 수 있습니다'}</span>
                    )}
                    {livePending && <span className="badge-warn">→ r003 재시작 후 적용 (실행 중: {liveRunning ? '실제 매매' : '꺼짐'})</span>}
                  </div>
                  {(liveOn || liveRunning) && liveLimits && (
                    <div className="tk-opt tk-opt-in">
                      <EdgeLiveLimitsEdit st={liveLimits} disabled={busy || !connected || modeBusy} />
                    </div>
                  )}
                  {liveSummary && (liveOn || liveRunning || liveSummary.held.length > 0 || liveSummary.openBuys.length > 0) && (
                    <div className="tk-opt tk-opt-in">
                      <span className={liveSummary.unpricedSells > 0 ? 'badge-warn' : 'muted small'} title="Edge AI 실제 매매의 오늘 현황 (서버 체결 장부 기준, 상태 조회 때마다 갱신)">
                        현황: {edgeLiveSummaryText(liveSummary, liveLimits ? liveLimits.entries.running ?? liveLimits.entries.effective : null)}
                      </span>
                    </div>
                  )}
                  <div className="tk-opt">
                    <label className="check" title="실제 주문 없이 가상 매수 / 청산을 기록 (가상 매매 탭, 서버 EDGE_ONLY_LOG_ONLY 의 반대). r003 재시작 후 적용">
                      <input
                        type="checkbox"
                        checked={!liveOn && paperBuys}
                        disabled={busy || !connected || modeBusy || !logOnly || liveOn}
                        onChange={(e) => void setPaperBuys(e.target.checked)}
                      />{' '}
                      가상 매매
                    </label>
                    <span className="muted small">
                      {liveOn
                        ? '실제 매매 중에는 가상 매수 없이 신호 후보만 기록합니다'
                        : paperBuys
                          ? '실제 주문 없이 가상 매매 탭에 기록'
                          : '꺼짐 - 신호가 뜬 봉을 후보로만 기록'}
                      {livePending ? ` (r003 재시작 전까지는 실행 중 설정: ${liveRunning ? '실제 매매 - 후보만 기록' : '가상 매매 설정대로'})` : ''}
                    </span>
                    {!liveOn && paperBuysPending && <span className="badge-warn">→ r003 재시작 후 적용 (실행 중: {paperBuysRunning ? '가상 매매' : '후보만'})</span>}
                  </div>
                </div>
              )}
            </div>
          )}
          {rb && (
            <div className="tk-head tk-gap">
              <b>반등 알림</b>
            </div>
          )}
          {rb && (
            <div className="tk">
              <label className="check tk-main" title="관심 종목의 반등 조건 감시 + 팝업 / 카카오 알림 (이 PC 의 앱에서 동작, 바로 적용)">
                <input type="checkbox" checked={!!reboundOn} disabled={reboundOn === null} onChange={(e) => void setKind('rebound', e.target.checked)} />{' '}
                {TRADE_KIND_KO.rebound}
              </label>
              <span className="muted small">관심 종목 반등 감시 + 팝업 / 카카오 알림 · 바로 적용 (세부 설정은 반등감시 창)</span>
            </div>
          )}
          <div className="tk-head tk-gap">
            <b>매도 설정</b>
          </div>
          <div className="tk tk-row">
          {/* [2026-11-12] automatic sells: everything / only the net -2% stop / none (the user sells by hand) */}
          <label title="r003 재시작 후 적용됩니다. '손절만'은 수수료+거래세 포함 순손익 -2%에서만 자동 매도하고, '끔'은 봇이 어떤 매도도 하지 않습니다. 두 경우 모두 당일 청산을 하지 않습니다.">
            자동 매도{' '}
            <select value={autoSellMode} disabled={busy || !autoSell || !connected || modeBusy} onChange={(e) => void setAutoSell(e.target.value as AutoSellMode)}>
              {AUTO_SELL_MODES.map((m) => (
                <option key={m} value={m}>
                  {AUTO_SELL_KO[m]}
                </option>
              ))}
            </select>
          </label>
          {autoSell ? (
            <span className={autoSellMode === 'all' ? 'muted small' : 'badge-warn'}>
              {autoSellPending ? `→ r003 재시작 후 적용 (실행 중: ${AUTO_SELL_KO[autoSellRunning]})` : autoSellMode === 'all' ? '' : '매도는 직접 해야 합니다 (당일 청산 없음)'}
            </span>
          ) : (
            strategy && <span className="muted small">자동 매도 선택은 서버 코드 갱신 후 쓸 수 있습니다</span>
          )}
          {/* [2026-11-13] which quantity the automatic sells may sell: the bot's own buys only / hand-bought (MTS) shares too */}
          {sellScope && (
            <label title="r003 재시작 후 적용됩니다. '봇 매수분만'은 봇이 매수한 수량만 자동 매도하고 MTS 등으로 직접 매수한 수량은 남깁니다. '직접 매수분 포함'은 봇이 관리하는 종목의 계좌 수량 전부를 매도합니다. 어느 쪽이든 전일 이월·감시목록 밖 보유분은 매도하지 않습니다.">
              자동 매도 대상{' '}
              <select
                value={sellScopeMode}
                disabled={busy || !connected || modeBusy || autoSellMode === 'off'}
                onChange={(e) => void setSellScope(e.target.value as AutoSellScope)}
              >
                {AUTO_SELL_SCOPES.map((m) => (
                  <option key={m} value={m}>
                    {AUTO_SELL_SCOPE_KO[m]}
                  </option>
                ))}
              </select>
            </label>
          )}
          {sellScope && (sellScopePending || sellScopeMode === 'all') && (
            <span className={sellScopePending ? 'badge-warn' : 'muted small'}>
              {sellScopePending ? `→ r003 재시작 후 적용 (실행 중: ${AUTO_SELL_SCOPE_KO[sellScopeRunning]})` : '직접 매수한 수량도 자동 매도됩니다'}
            </span>
          )}
          </div>
        </div>
        <div className="card-row">
          {edgeModel !== undefined && <EdgeModelLine m={edgeModel} tradeMode={tradeMode} />}
          {nPending > 0 && st?.running && (
            <span className="badge-warn" title="서비스 시작 이후 바뀐 파일이 있습니다">
              ⚠ 미적용 변경 {nPending}건 → 재시작 필요
            </span>
          )}
          <span className="push-right muted small" title={`상태 기준 시각 ${isoShow(st?.generated_at)} (KST)`}>
            {noteRight}
          </span>
        </div>
        {(msg || (control.lastAction && !busy) || control.error || st?.external_r003?.length || (st && !st.running && st.override_count > 0)) && (
          <div className="card-body notes">
            {msg && <p className="err-text">{msg}</p>}
            {control.lastAction && !busy && <p className={control.lastAction.ok ? 'muted small' : 'err-text small'}>마지막 동작: {control.lastAction.message}</p>}
            {control.error && <p className="err-text small">상태 조회 오류: {control.error}</p>}
            {!!st?.external_r003?.length && (
              <div className="callout danger">
                <b>외부 r003 감지</b> -{' '}
                {isLocal ? '감시 프로세스 밖에서 실행된 r003입니다 (콘솔 등). [정지]를 누르면 상태를 저장하고 끝내도록 요청합니다.' : '서버의 screen 세션에서 r003을 종료한 뒤 앱에서 실행하세요.'}
                <ul>
                  {st.external_r003.map((p) => (
                    <li key={p.pid} className="mono">
                      PID {p.pid} · {p.cmdline}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {st && !st.running && st.override_count > 0 && <p className="callout small">봇이 꺼져 있습니다. 저장된 오버라이드 {st.override_count}건은 다음 실행 때 적용됩니다.</p>}
            {svc?.sub_state === 'auto-restart' && <p className="warn-text small">크래시 후 재시작 대기 중</p>}
            {svc?.active_state === 'failed' && <p className="warn-text small">마지막 실행 실패 (result={svc.result})</p>}
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <div>
            <span className="title">미적용 변경</span>
            <span className="count">{pendingRows.length}건</span>
          </div>
          <div className="card-actions">
            <button disabled={busy || !connected || !overrides.length} onClick={() => void reset()}>
              기본값으로 되돌리기
            </button>
          </div>
        </div>
        <div className="card-body">
          {(st?.override_errors?.length || 0) > 0 && (
            <div className="callout warn small">
              서버(r001)가 거부한 오버라이드 항목:
              <ul>
                {st!.override_errors.map((e) => (
                  <li key={e}>{e}</li>
                ))}
              </ul>
            </div>
          )}
          {pendingRows.length ? (
            <table className="grid compact">
              <tbody>
                {pendingRows.map((r) => (
                  <tr key={r.k} className="nohover">
                    <td>{r.pending ? <i className="pend-dot">미적용</i> : <i className="ovr-mark">{st?.running ? '◆ 적용 중' : '◆ 저장됨'}</i>}</td>
                    <td className="mono wrap">{r.text}</td>
                    <td>
                      {r.key && (
                        <button className="mini" disabled={busy || !connected} onClick={() => void unset(r.key!)}>
                          기본값으로
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">재시작을 기다리는 변경이 없습니다.</p>
          )}
          <p className="muted small">설정값은 파이프라인 탭에서 노드를 눌러 바꾸거나, 설정/연결 탭의 "고급: 전체 설정값"에서 바꿉니다.</p>
        </div>
      </section>

      <section className="card">
        <div className="card-head">
          <div>
            <span className="title">최근 변경 이력</span>
            <span className="count">{history.length}건</span>
          </div>
        </div>
        {histErr && <p className="err-text small card-body">{histErr}</p>}
        <div className="table-wrap short">
          <table className="hist">
            <tbody>
              {history.map((h, i) => (
                <tr key={`${h.ts}-${i}`}>
                  <td className="t" title={isoShow(h.ts)}>
                    {timeOnly(h.ts)}
                  </td>
                  <td className="a">{ACTION_LABEL[h.action] || h.action}</td>
                  <td className="b muted">{h.by}</td>
                  <td className="c mono wrap">{historyText(h)}</td>
                </tr>
              ))}
              {!history.length && (
                <tr>
                  <td colSpan={4} className="muted center">
                    이력 없음
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;

/** [2026-11-12] one line: which Edge AI model the bot uses now, what it was trained on and how it did out of sample */
function EdgeModelLine({ m, tradeMode }: { m: EdgeModelInfo | null; tradeMode: TradeMode }) {
  if (!m) return <span className="muted small">Edge AI 모델: 배포된 모델 없음 (runtime/edge_ai/current_meta.json)</span>;
  const use = tradeMode === 'edge' ? '매수 신호로 사용 중' : tradeMode === 'rules+paper' ? '매수 필터(_020) + 가상 매매에 사용 중' : '매수 필터(_020)에 사용 중';
  const detail = [
    `모델 ${m.modelId} (생성 ${m.created.replace('T', ' ')})`,
    `학습 대상: ${edgeLabelKo(m)}${m.horizon && m.target === 'y_tp' ? ` · 당일 장 마감까지 관찰` : ''}`,
    `데이터셋 ${m.dataset || '-'} · ${m.nDays ?? '?'}일 · 최종 학습 구간 ${m.trainRange || '-'}`,
    `확률 기준 ${m.threshold === null ? '-' : m.threshold.toFixed(3)} · 검증 AUC ${m.aucOos === null ? '-' : m.aucOos.toFixed(3)}`,
    m.kept && m.base ? `검증 구간 순손익: 통과 ${m.kept.n.toLocaleString()}건 ${m.kept.meanNet.toFixed(3)}% / 전체 ${m.base.n.toLocaleString()}건 ${m.base.meanNet.toFixed(3)}%` : '',
    m.runKept && m.runBase
      ? `손절 전 도달률(통과 / 전체): +2% ${pct1(m.runKept.tp2)} / ${pct1(m.runBase.tp2)} · +3% ${pct1(m.runKept.tp3)} / ${pct1(m.runBase.tp3)} · +5% ${pct1(m.runKept.tp5)} / ${pct1(m.runBase.tp5)} · 손절 ${pct1(m.runKept.stopped)} / ${pct1(m.runBase.stopped)}`
      : '',
    m.promoted ? '승격됨: 검증 구간에서 통과 진입의 순손익이 0보다 컸습니다.' : '승격 안 됨: 검증 구간에서 통과 진입의 순손익이 0보다 크지 않았습니다.',
    '봇은 이 파일이 바뀌면 1분 안에 새 모델을 읽습니다 (재시작 불필요).',
  ]
    .filter(Boolean)
    .join('\n');
  return (
    <span className="muted small edge-model" title={detail}>
      Edge AI 모델 <b className="mono">{m.modelId}</b> · {edgeLabelKo(m)} · 기준 {m.threshold === null ? '-' : m.threshold.toFixed(3)}
      {m.runKept ? ` · +2% 도달 ${pct1(m.runKept.tp2)}` : ''} · <span className={m.promoted ? '' : 'warn-text'}>{m.promoted ? '승격됨' : '승격 안 됨'}</span> · {use}
    </span>
  );
}

const EXIT_KO: Record<PaperExitKind, string> = { tp: '익절', sl: '손절', horizon: '보유 시간' };
/** "+2%" / "-1.77%" / "400분" */
const exitText = (k: PaperExitKind, v: number | null | undefined): string =>
  v === null || v === undefined ? '?' : k === 'tp' ? `+${v}%` : k === 'sl' ? `-${v}%` : `${v}분`;

/** [2026-11-25] the Edge AI exits (server r001 EDGE_ONLY_TP_PCT / EDGE_ONLY_SL_PCT / EDGE_ONLY_HORIZON_MIN), one place for the paper
 *  runner (r022) and the edge strategy's real sells (r006 _e10~_e12). An empty box = 0 = paper: the model's label (the placeholder),
 *  edge: the server's EDGE_TP_PCT / EDGE_SL_PCT / EDGE_MAX_HOLD_MINUTES. Paper: new paper buys after the next r003 start (open paper
 *  positions keep theirs); edge: every evaluation after the restart, held positions too. real = the server runs the edge strategy */
function PaperExitEdit({ st, model, real, disabled }: { st: PaperExitState; model: EdgeModelInfo | null; real: boolean; disabled: boolean }) {
  const fromState = (): Record<PaperExitKind, string> => ({
    tp: st.tp.effective > 0 ? String(st.tp.effective) : '',
    sl: st.sl.effective > 0 ? String(st.sl.effective) : '',
    horizon: st.horizon.effective > 0 ? String(st.horizon.effective) : '',
  });
  const [text, setText] = useState(fromState);
  const [saving, setSaving] = useState(false);
  const stKey = PAPER_EXIT_KINDS.map((k) => st[k].effective).join(',');
  useEffect(() => setText(fromState()), [stKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const vals = { tp: paperExitInput('tp', text.tp), sl: paperExitInput('sl', text.sl), horizon: paperExitInput('horizon', text.horizon) };
  const bad = PAPER_EXIT_KINDS.filter((k) => vals[k] === null);
  const changed = !bad.length && PAPER_EXIT_KINDS.some((k) => vals[k] !== st[k].effective);
  const pending = PAPER_EXIT_KINDS.some((k) => st[k].running !== null && (st[k].pending || st[k].running !== st[k].effective));
  const modelOf = (k: PaperExitKind) => (model ? model[k] : null);
  /** what a stored value means: the set value, or the model's */
  const meaning = (k: PaperExitKind, v: number) =>
    v > 0 ? exitText(k, v) : real ? '서버 기본값 (EDGE_TP_PCT 등)' : `모델 값 ${exitText(k, modelOf(k))}`;
  const save = async () => {
    if (!changed) return;
    const v = vals as Record<PaperExitKind, number>;
    const lines = PAPER_EXIT_KINDS.map((k) => `${EXIT_KO[k]}: ${meaning(k, v[k])}`).join('\n');
    const custom = PAPER_EXIT_KINDS.some((k) => v[k] > 0);
    const note = custom ? '\n\n모델 확률은 모델 라벨 기준(익절/손절/시간)으로 학습된 값이라, 기준을 바꾸면 확률의 뜻과 어긋납니다.' : '';
    if (
      !window.confirm(
        real
          ? `Edge AI 청산 기준 (실제 주문)\n\n${lines}${note}\n\nEdge AI 실제 매도 기준이 바뀝니다. r003 재시작 후부터 이미 보유한 종목에도 적용됩니다.\n${EDGE_LIVE_EXIT_NOTE}\n(Edge AI 단독 방식 edge 는 자동 매도가 전체일 때만 적용) 바꿀까요?`
          : `Edge AI 청산 기준 (가상 매매)\n\n${lines}${note}\n\n실제 주문은 없습니다. r003 재시작 후 새 가상 매수부터 적용되고, 이미 있는 가상 보유는 매수 때 정한 기준 그대로입니다. 가상 매매와 Edge AI 실주문 방식이 같은 값을 씁니다. 바꿀까요?`,
      )
    )
      return;
    setSaving(true);
    try {
      const r = await window.api.ctlSet(paperExitChanges(v));
      if (!r.ok) window.alert(`Edge AI 청산 기준을 바꾸지 못했습니다:${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setSaving(false);
    }
  };
  const box = (k: PaperExitKind) => (
    <input
      className="paper-exit-in"
      type="text"
      inputMode="decimal"
      value={text[k]}
      placeholder={real ? '기본' : modelOf(k) !== null ? String(modelOf(k)) : '모델'}
      disabled={disabled || saving}
      aria-invalid={vals[k] === null}
      onChange={(e) => setText((t) => ({ ...t, [k]: e.target.value }))}
      onKeyDown={(e) => e.key === 'Enter' && void save()}
    />
  );
  return (
    <span
      className="paper-exit"
      title={`Edge AI 청산 기준 - 가상 매매와 Edge AI 실주문 공통 (서버 EDGE_ONLY_TP_PCT / EDGE_ONLY_SL_PCT / EDGE_ONLY_HORIZON_MIN). 가격 기준(수수료 제외). 빈칸 = 가상: 모델 값(흐린 숫자), 실주문: 서버 기본(+2.0% / -1.7% / 60분). 규칙 매매에는 쓰지 않습니다. 익절·손절 0~30%, 보유 시간 0~720분. r003 재시작 후 적용`}
    >
      청산 기준 익절 +{box('tp')}% 손절 -{box('sl')}% 보유 {box('horizon')}분
      <button className="mini" disabled={disabled || saving || !changed} onClick={() => void save()}>
        적용
      </button>
      {bad.length > 0 && <span className="badge-warn">{bad.map((k) => EXIT_KO[k]).join(', ')} 값 확인</span>}
      {pending && (
        <span className="badge-warn">
          → r003 재시작 후 적용 (실행 중: {PAPER_EXIT_KINDS.map((k) => meaning(k, st[k].running ?? st[k].effective)).join(' / ')})
        </span>
      )}
    </span>
  );
}

/** [2026-11-25] the limits of the Edge AI live orders (server r001 EDGE_LIVE_ORDER_KRW / EDGE_LIVE_MAX_POSITIONS /
 *  EDGE_LIVE_MAX_ENTRIES_PER_DAY / EDGE_LIVE_DAILY_LOSS_KRW; r016 checks the same ranges). Whole numbers, applied after the next
 *  r003 start. At a limit the server stops new edge buys for the day - what the edge strategy holds is still closed */
function EdgeLiveLimitsEdit({ st, disabled }: { st: EdgeLiveLimits; disabled: boolean }) {
  const fromState = (): Record<EdgeLiveLimitKind, string> => ({
    order: String(st.order.effective),
    positions: String(st.positions.effective),
    entries: String(st.entries.effective),
    loss: String(st.loss.effective),
  });
  const [text, setText] = useState(fromState);
  const [saving, setSaving] = useState(false);
  const stKey = EDGE_LIVE_LIMIT_KINDS.map((k) => st[k].effective).join(',');
  useEffect(() => setText(fromState()), [stKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const vals = Object.fromEntries(EDGE_LIVE_LIMIT_KINDS.map((k) => [k, edgeLiveLimitInput(k, text[k])])) as Record<EdgeLiveLimitKind, number | null>;
  const bad = EDGE_LIVE_LIMIT_KINDS.filter((k) => vals[k] === null);
  const changed = !bad.length && EDGE_LIVE_LIMIT_KINDS.some((k) => vals[k] !== st[k].effective);
  const pending = EDGE_LIVE_LIMIT_KINDS.some((k) => st[k].running !== null && (st[k].pending || st[k].running !== st[k].effective));
  const show = (k: EdgeLiveLimitKind, v: number) => `${EDGE_LIVE_LIMIT_KO[k]} ${v.toLocaleString('ko-KR')}${EDGE_LIVE_LIMIT_UNIT[k]}`;
  const save = async () => {
    if (!changed) return;
    const v = vals as Record<EdgeLiveLimitKind, number>;
    const lines = EDGE_LIVE_LIMIT_KINDS.map((k) => show(k, v[k])).join('\n');
    if (!window.confirm(`Edge AI 실제 주문 한도\n\n${lines}\n\n한도에 닿으면 그날 Edge 신규 매수를 멈춥니다(보유분 청산은 계속).\nr003 재시작 후부터 적용됩니다. 바꿀까요?`)) return;
    setSaving(true);
    try {
      const r = await window.api.ctlSet(edgeLiveLimitChanges(v));
      if (!r.ok) window.alert(`Edge AI 실제 주문 한도를 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
    } finally {
      setSaving(false);
    }
  };
  const box = (k: EdgeLiveLimitKind) => (
    <input
      className={k === 'order' || k === 'loss' ? 'paper-exit-in edge-live-in' : 'paper-exit-in'}
      type="text"
      inputMode="numeric"
      value={text[k]}
      disabled={disabled || saving}
      aria-invalid={vals[k] === null}
      title={`${EDGE_LIVE_LIMIT_RANGE[k][0].toLocaleString('ko-KR')} ~ ${EDGE_LIVE_LIMIT_RANGE[k][1].toLocaleString('ko-KR')}${EDGE_LIVE_LIMIT_UNIT[k]}`}
      onChange={(e) => setText((t) => ({ ...t, [k]: e.target.value }))}
      onKeyDown={(e) => e.key === 'Enter' && void save()}
    />
  );
  return (
    <span className="paper-exit" title="Edge AI 실제 주문 한도 (서버 EDGE_LIVE_*). 한도에 닿으면 그날 Edge 신규 매수 중단. r003 재시작 후 적용">
      실제 주문 한도 1회 {box('order')}원 동시 {box('positions')}종목 하루 {box('entries')}회 손실 {box('loss')}원
      <button className="mini" disabled={disabled || saving || !changed} onClick={() => void save()}>
        적용
      </button>
      {bad.length > 0 && <span className="badge-warn">{bad.map((k) => EDGE_LIVE_LIMIT_KO[k]).join(', ')} 값 확인</span>}
      {pending && (
        <span className="badge-warn">
          → r003 재시작 후 적용 (실행 중: {EDGE_LIVE_LIMIT_KINDS.map((k) => show(k, st[k].running ?? st[k].effective)).join(' / ')})
        </span>
      )}
    </span>
  );
}

/** [2026-11-25] the rebound watcher's on / off of AutoBotHtsH (its preload has these; AutoBotH has no rebound watcher -> null) */
type ReboundApi = { reboundStatus: () => Promise<{ enabled: boolean }>; reboundEnable: (on: boolean) => Promise<{ enabled: boolean }> };
function reboundApi(): ReboundApi | null {
  // AutoBotHtsH's preload exposes them on window.hts (Codex review)
  const a = (window as unknown as { hts?: Partial<ReboundApi> }).hts;
  return a && typeof a.reboundStatus === 'function' && typeof a.reboundEnable === 'function' ? (a as ReboundApi) : null;
}
