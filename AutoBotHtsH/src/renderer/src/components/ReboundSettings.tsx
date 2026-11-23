// Update log
// - [2026-11-21] type=feat (☰ > 반등 감시 설정 (user 2026-11-21): the rebound watcher is set here instead of editing config.json by hand -
//   on / off, the watchlist tab to watch (pick one or make it), the symbols of that tab (add / remove, same list as the watchlist
//   window), every threshold of the detector, the alert, polling and session hours, the indicator periods. 저장 writes
//   %APPDATA%\<app>\rebound\config.json through the main process (values out of range fall back to the defaults there); the
//   watcher uses it from the next minute. Also: defaults, test popup, open the folder)
// - [2026-11-22] type=feat (Tabs: 설정 / 히스토리 (alerts shown so far, ReboundPanels.HistoryTab) / 카카오톡 (send alerts to the user's own
//   KakaoTalk, ReboundPanels.KakaoTab). "팝업 유지 시간" = alert.autoCloseSec)

import { useEffect, useMemo, useState } from 'react';
import { newGroupId, useStore, WATCH_MAX } from '../store';
import { SymbolSearch } from './SymbolSearch';
import { REBOUND_DEFAULTS, type ReboundConfig } from '../../../shared/rebound';
import type { ReboundStatus } from '../../../preload/index';
import { HistoryTab, KakaoTab } from './ReboundPanels';

type Tab = 'settings' | 'history' | 'kakao';

type Kind = 'num' | 'bool' | 'method';
interface Field {
  path: string;
  label: string;
  hint?: string;
  kind?: Kind;
  unit?: string;
}
const SECTIONS: { title: string; note?: string; fields: Field[] }[] = [
  {
    title: 'A 급락 봉',
    fields: [
      { path: 'plunge.window', label: '찾는 범위', unit: '봉', hint: '최근 몇 봉 안에서 급락 봉을 찾을지' },
      { path: 'plunge.dropPct', label: '1봉 하락률', unit: '% 이하', hint: '전봉 종가 대비 (음수)' },
      { path: 'plunge.wickRatio', label: '아랫꼬리 / 몸통', unit: '배 이상', hint: '하락률 대신 인정하는 긴 아랫꼬리' },
      { path: 'plunge.wickMinPct', label: '아랫꼬리 최소 길이', unit: '% (가격 대비)' },
      { path: 'plunge.volMult', label: '거래량 배수', unit: '배 이상', hint: '앞 봉들의 평균 거래량 대비' },
      { path: 'plunge.volWindow', label: '평균 거래량 범위', unit: '봉' },
      { path: 'plunge.maxFallPct', label: '낙폭 한도 (악재성 급락 제외)', unit: '% (0 = 제한 없음)', hint: '급락 봉 앞 최고가 -> 급락 봉 저가 낙폭이 이보다 크면 감지하지 않음' },
      { path: 'plunge.fallWindow', label: '낙폭을 재는 범위', unit: '봉' },
      { path: 'plunge.gapPct', label: '갭 하락 시가 인정', unit: '% 이하 (0 = 끔)', hint: '급락 봉이 없을 때, 당일 첫 봉 시가가 전봉 종가보다 이만큼 낮게 시작했으면 급락 선행으로 인정 (음수). 급락 봉은 시가 이후 최저가 봉' },
    ],
  },
  {
    title: 'B~F 돌파 조건',
    fields: [
      { path: 'lookback', label: '돌파 인정 범위', unit: '봉', hint: 'RSI / MACD / DI / MA5-MA20 돌파가 최근 몇 봉 안에 있어야 하는지' },
      { path: 'recent', label: '유지 확인 범위', unit: '봉' },
      { path: 'recentMin', label: '그중 위에 있어야 하는 봉 수', unit: '봉 이상', hint: 'RSI > 시그널, DI+ > DI-' },
      { path: 'requireCurrentAbove', label: '현재 봉도 위에 있어야 함', kind: 'bool' },
      { path: 'rsi.min', label: 'RSI 하한' },
      { path: 'rsi.max', label: 'RSI 상한', hint: '과매수 전' },
      { path: 'macd.oscRising', label: 'MACD 오실레이터 연속 확대', unit: '봉' },
      { path: 'angle.min', label: 'MA5 각도', unit: '° 이상' },
      { path: 'angle.k', label: '각도 배율 K', hint: '각도 = atan(1봉당 % 변화 x K)' },
      { path: 'angle.lag', label: '각도 비교 간격', unit: '봉 전' },
      { path: 'minBars', label: '판정에 필요한 봉 수', unit: '봉', hint: '지표 안정화' },
    ],
  },
  {
    title: '가점',
    fields: [
      { path: 'bonus.minScore', label: '알림 기준 점수', unit: '점 이상 (6점 만점)' },
      { path: 'bonus.adxTroughBars', label: 'ADX 바닥 인정 범위', unit: '봉 안' },
      { path: 'bonus.adxDropMin', label: 'ADX 앞 고점 대비 하락폭', unit: '이상' },
      { path: 'bonus.maNearPct', label: 'MA60 / MA120 근접', unit: '% 이내' },
      { path: 'bonus.macdZeroPct', label: 'MACD 0선 근처', unit: '% (가격 대비)' },
      { path: 'bonus.macdZeroBars', label: 'MACD 0선 돌파 인정 범위', unit: '봉 안' },
    ],
  },
  {
    title: '장 경계 제외',
    note: '각 운영 시간대의 처음 / 마지막 구간은 감지하지 않고 급락 봉으로도 치지 않습니다 (0 = 끔).',
    fields: [
      { path: 'edge.openMin', label: '시작 후', unit: '분' },
      { path: 'edge.closeMin', label: '끝나기 전', unit: '분' },
      { path: 'edge.preOpenMin', label: 'NXT 프리마켓 시작 후', unit: '분', hint: '프리마켓(08:00~08:50)은 짧아서 따로 둡니다' },
      { path: 'edge.preCloseMin', label: 'NXT 프리마켓 끝나기 전', unit: '분' },
    ],
  },
  {
    title: '알림',
    fields: [
      { path: 'alert.cooldownMin', label: '같은 종목 다시 알림까지', unit: '분' },
      { path: 'alert.sound', label: '알림음', kind: 'bool' },
      { path: 'alert.autoCloseSec', label: '팝업 유지 시간', unit: '초 (0 = 닫지 않음)', hint: '이 시간이 지나 저절로 닫힌 알람은 미확인으로 남아 제목 줄에 개수가 표시됩니다' },
      { path: 'alert.maxPopups', label: '화면에 두는 팝업 수', unit: '개' },
    ],
  },
  {
    title: '조회',
    fields: [
      { path: 'poll.delaySec', label: '매분 조회 시점', unit: '초' },
      { path: 'poll.days', label: '분봉 조회 일수', unit: '일', hint: '전일 분봉으로 지표를 안정시킴' },
      { path: 'poll.gapMs', label: '종목 사이 간격', unit: 'ms' },
      { path: 'poll.tabReloadSec', label: '탭 다시 읽기', unit: '초마다' },
      { path: 'poll.retries', label: '조회 재시도', unit: '회' },
      { path: 'poll.backoffMs', label: '재시도 첫 대기', unit: 'ms (매번 2배)' },
    ],
  },
  {
    title: '지표 (차트 지표 설정과 같게)',
    fields: [
      { path: 'indicator.rsiPeriod', label: 'RSI 기간' },
      { path: 'indicator.rsiSignal', label: 'RSI 시그널' },
      { path: 'indicator.rsiMethod', label: 'RSI 방식', kind: 'method' },
      { path: 'indicator.macdFast', label: 'MACD 단기' },
      { path: 'indicator.macdSlow', label: 'MACD 장기' },
      { path: 'indicator.macdSignal', label: 'MACD 시그널' },
      { path: 'indicator.adxPeriod', label: 'DMI / ADX 기간' },
      { path: 'indicator.adxMethod', label: 'DMI / ADX 방식', kind: 'method' },
      { path: 'indicator.bbPeriod', label: '볼린저 기간' },
      { path: 'indicator.bbMult', label: '볼린저 배수' },
      { path: 'indicator.ichiTenkan', label: '일목 전환선' },
      { path: 'indicator.ichiKijun', label: '일목 기준선' },
      { path: 'indicator.ichiSenkouB', label: '일목 선행스팬 B' },
      { path: 'indicator.ichiShift', label: '일목 선행 이동', unit: '봉' },
    ],
  },
];

type Obj = Record<string, unknown>;
const getAt = (o: unknown, path: string): unknown => path.split('.').reduce<unknown>((a, k) => (a && typeof a === 'object' ? (a as Obj)[k] : undefined), o);
function setAt<T>(o: T, path: string, v: unknown): T {
  const keys = path.split('.');
  const out = { ...(o as Obj) };
  let cur = out;
  keys.slice(0, -1).forEach((k) => {
    cur[k] = { ...(cur[k] as Obj) };
    cur = cur[k] as Obj;
  });
  cur[keys[keys.length - 1]] = v;
  return out as T;
}

const STATE: Record<ReboundStatus['state'], string> = { off: '꺼짐', rest: '장 시간 밖 (대기)', login: 'KIS 로그인 대기', notab: '탭 없음', run: '감시 중' };
const SESSION_LABEL: Record<'krx' | 'nxtPre' | 'nxtAfter', string> = { krx: 'KRX 정규장', nxtPre: 'NXT 프리마켓', nxtAfter: 'NXT 애프터마켓' };

function Names({ codes, onRemove }: { codes: string[]; onRemove: (code: string) => void }) {
  const [names, setNames] = useState<Record<string, string>>({});
  useEffect(() => {
    let live = true;
    void Promise.all(codes.map((c) => window.hts.symbol(c).then((m) => [c, m?.name || ''] as const))).then((r) => live && setNames(Object.fromEntries(r)));
    return () => {
      live = false;
    };
  }, [codes.join()]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!codes.length) return <p className="muted small">종목이 없습니다. 아래에서 추가하세요.</p>;
  return (
    <div className="rb-codes">
      {codes.map((c) => (
        <span key={c} className="rb-code">
          {names[c] || c} <i>{c}</i>
          <button type="button" title="이 탭에서 빼기" onClick={() => onRemove(c)}>
            ×
          </button>
        </span>
      ))}
    </div>
  );
}

export function ReboundSettings({ onClose, initialTab = 'settings' }: { onClose: () => void; initialTab?: Tab }) {
  const [tab, setTab] = useState<Tab>(initialTab);
  const watch = useStore((s) => s.watch);
  const setWatch = useStore((s) => s.setWatch);
  const [cfg, setCfg] = useState<ReboundConfig | null>(null);
  const [saved, setSaved] = useState('');
  const [st, setSt] = useState<ReboundStatus | null>(null);
  const [msg, setMsg] = useState('');

  useEffect(() => {
    void window.hts.reboundConfig().then((c) => (setCfg(c), setSaved(JSON.stringify(c))));
    void window.hts.reboundStatus().then(setSt);
    const off = window.hts.onReboundStatus(setSt);
    return () => void off();
  }, []);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', esc);
    return () => window.removeEventListener('keydown', esc);
  }, [onClose]);

  const tabName = cfg?.tab.trim() || '';
  const group = useMemo(() => watch.find((g) => g.name.trim().toLowerCase() === tabName.toLowerCase()), [watch, tabName]);
  if (!cfg) return null;
  const dirty = JSON.stringify(cfg) !== saved;
  const set = (path: string, v: unknown) => (setCfg(setAt(cfg, path, v)), setMsg(''));

  const save = async () => {
    const r = await window.hts.reboundSaveConfig(cfg);
    setCfg(r.config);
    setSaved(JSON.stringify(r.config));
    setSt(r.status);
    setMsg(JSON.stringify(r.config) === JSON.stringify(cfg) ? '저장했습니다. 다음 분부터 적용됩니다.' : '저장했습니다. 범위를 벗어난 값은 기본값으로 바뀌었습니다.');
  };
  // the symbol list is the watchlist tab itself: changes are kept at once (like the watchlist window)
  const editCodes = (codes: string[]) => {
    if (!group) return;
    setWatch(watch.map((g) => (g.id === group.id ? { ...g, codes } : g)));
    void window.hts.reboundRefresh();
  };
  const makeTab = () => {
    if (!tabName) return;
    setWatch([...watch, { id: newGroupId(), name: tabName, codes: [] }]);
    void window.hts.reboundRefresh();
  };

  const input = (f: Field) => {
    const v = getAt(cfg, f.path);
    if (f.kind === 'bool') return <input type="checkbox" checked={!!v} onChange={(e) => set(f.path, e.target.checked)} />;
    if (f.kind === 'method')
      return (
        <select value={String(v)} onChange={(e) => set(f.path, e.target.value)}>
          <option value="mts">MTS 방식 (차트 기본)</option>
          <option value="wilder">Wilder 방식</option>
        </select>
      );
    const n = v as number;
    return <input type="number" step="any" value={Number.isFinite(n) ? n : ''} onChange={(e) => set(f.path, e.target.value === '' ? NaN : Number(e.target.value))} />;
  };
  const changed = (f: Field) => getAt(cfg, f.path) !== getAt(REBOUND_DEFAULTS, f.path);

  return (
    <div className="login-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="rb-dialog">
        <header>
          <h2>바닥 반등 초입 감시 설정</h2>
          <span className="muted small">감지·알림 전용 - 주문은 내지 않습니다</span>
          <button type="button" className="mini" onClick={onClose} title="닫기 (Esc)">
            ✕
          </button>
        </header>
        <nav className="rb-tabs seg">
          {([['settings', '설정'], ['history', '히스토리'], ['kakao', '카카오톡']] as [Tab, string][]).map(([k, label]) => (
            <button key={k} type="button" className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>
              {label}
            </button>
          ))}
        </nav>
        {tab === 'history' && <HistoryTab />}
        {tab === 'kakao' && <KakaoTab enabled={cfg.alert.kakao} onEnabled={(on) => set('alert.kakao', on)} />}
        <div className="rb-body" style={tab === 'settings' ? undefined : { display: 'none' }}>
          <section className="rb-top">
            <label className="check rb-onoff">
              <input type="checkbox" checked={cfg.enabled} onChange={(e) => set('enabled', e.target.checked)} /> <b>감시 켜기</b>
            </label>
            <span className="muted">
              현재 상태: <b>{st ? STATE[st.enabled ? st.state : 'off'] : '-'}</b>
              {st?.state === 'run' ? ` · ${st.codes}종목` : ''}
              {st?.lastPass ? ` · 마지막 순회 ${new Date(st.lastPass).toLocaleTimeString('ko-KR', { hour12: false })}` : ''}
              {st ? ` · 알림 ${st.alerts}회` : ''}
            </span>
            {st?.note && <span className="rb-warn">{st.note}</span>}
          </section>

          <section>
            <h3>감시 대상</h3>
            <div className="rb-row">
              <label>
                관심종목 탭 이름
                <input type="text" value={cfg.tab} maxLength={20} onChange={(e) => set('tab', e.target.value)} />
              </label>
              <label>
                탭에서 고르기
                <select value={group ? group.name : ''} onChange={(e) => e.target.value && set('tab', e.target.value)}>
                  <option value="">{group ? '' : '(없는 탭)'}</option>
                  {watch.map((g) => (
                    <option key={g.id} value={g.name}>
                      {g.name} ({g.codes.length})
                    </option>
                  ))}
                </select>
              </label>
              {!group && tabName && (
                <button type="button" onClick={makeTab} title="관심종목에 이 이름의 빈 탭을 만듭니다">
                  "{tabName}" 탭 만들기
                </button>
              )}
            </div>
            {group ? (
              <>
                <p className="muted small">
                  "{group.name}" 탭의 종목 {group.codes.length}/{WATCH_MAX} - 여기서 바꾸면 관심종목 창에도 바로 반영됩니다.
                  {group.src ? ` 이 탭은 ${group.src} 파일에서 만든 탭이라 파일이 바뀌면 목록이 다시 맞춰집니다.` : ''}
                </p>
                <Names codes={group.codes} onRemove={(c) => editCodes(group.codes.filter((x) => x !== c))} />
                <div className="rb-add">
                  <SymbolSearch compact placeholder="종목 추가: 종목명 / 코드 / 초성" onPick={(c) => !group.codes.includes(c) && group.codes.length < WATCH_MAX && editCodes([...group.codes, c])} />
                </div>
              </>
            ) : (
              <p className="rb-warn">관심종목에 "{tabName}" 탭이 없습니다. 탭을 고르거나 만들어야 감시가 시작됩니다.</p>
            )}
          </section>

          <section>
            <h3>운영 시간 (평일, KST)</h3>
            <div className="rb-row">
              {(['krx', 'nxtPre', 'nxtAfter'] as const).map((k) => {
                const w = cfg.sessions[k];
                return (
                  <label key={k} className="rb-session">
                    <span>
                      {k !== 'krx' && <input type="checkbox" checked={!!w} onChange={(e) => set(`sessions.${k}`, e.target.checked ? REBOUND_DEFAULTS.sessions[k] : null)} />} {SESSION_LABEL[k]}
                    </span>
                    {w ? (
                      <span>
                        <input type="time" value={w[0].padStart(5, '0')} onChange={(e) => e.target.value && set(`sessions.${k}`, [e.target.value, w[1]])} /> ~{' '}
                        <input type="time" value={w[1].padStart(5, '0')} onChange={(e) => e.target.value && set(`sessions.${k}`, [w[0], e.target.value])} />
                      </span>
                    ) : (
                      <span className="muted">감시 안 함</span>
                    )}
                  </label>
                );
              })}
            </div>
            <p className="muted small">NXT 시간에는 NXT 거래 종목만 조회합니다.</p>
          </section>

          {SECTIONS.map((s) => (
            <section key={s.title}>
              <h3>{s.title}</h3>
              {s.note && <p className="muted small">{s.note}</p>}
              <div className="rb-grid">
                {s.fields.map((f) => (
                  <label key={f.path} className={`rb-field ${changed(f) ? 'changed' : ''}`} title={`${f.hint ? `${f.hint}\n` : ''}기본값: ${String(getAt(REBOUND_DEFAULTS, f.path))}`}>
                    <span>{f.label}</span>
                    {input(f)}
                    <em>{f.unit || ''}</em>
                  </label>
                ))}
              </div>
            </section>
          ))}
        </div>
        <footer>
          <button type="button" onClick={() => void window.hts.reboundOpenDir()} title="config.json 과 감지 기록(detections_YYYYMMDD.csv)이 있는 폴더">
            설정·기록 폴더
          </button>
          <button type="button" onClick={() => void window.hts.reboundTest()} title="예시 팝업을 띄웁니다">
            알람 테스트
          </button>
          <button type="button" onClick={() => (setCfg({ ...REBOUND_DEFAULTS, enabled: cfg.enabled, tab: cfg.tab }), setMsg('기본값을 불러왔습니다. 저장을 눌러야 적용됩니다.'))} title="탭 이름과 켜기/끄기는 그대로 두고 나머지를 기본값으로">
            기본값으로
          </button>
          <span className="rb-msg">{msg || (dirty ? '저장하지 않은 변경이 있습니다.' : '')}</span>
          <button type="button" className="primary" disabled={!dirty} onClick={() => void save()}>
            저장
          </button>
          <button type="button" onClick={onClose}>
            닫기
          </button>
        </footer>
      </div>
    </div>
  );
}
