// Update log
// - [2026-11-22] type=feat (Tabs of the rebound settings dialog (user 2026-11-22): HistoryTab - the alerts that were shown, newest
//   first (unread ones marked); a click shows everything the popup showed and marks it read; mark all read, delete, show the popup
//   again, open the symbol. KakaoTab - the user's Kakao Developers app (REST API key, client secret, redirect URI), Kakao login,
//   a test message, and the switch that sends each alert's summary to the user's own KakaoTalk)
// - [2026-11-22] type=feat (KakaoTab: "봇용 로그인 파일 만들기" - login files for the bot (~/.kakao, AutoBot g007) so the scan job
//   sends its picks (code + name) to KakaoTalk together with the result mail)

import { useEffect, useState } from 'react';
import { useStore } from '../store';
import { alertSummary, BONUS_KEYS, BONUS_LABEL, COND_KEYS, COND_LABEL, hhmm } from '../../../shared/rebound';
import type { KakaoState, ReboundHistoryItem } from '../../../preload/index';

const KST = 9 * 3600000;
const when = (t: number) => new Date(t + KST).toISOString().slice(5, 16).replace('T', ' ');
const num = (x: number | undefined, d = 1) => (x === undefined || !Number.isFinite(x) ? '-' : x.toLocaleString('ko-KR', { minimumFractionDigits: d, maximumFractionDigits: d }));

export function HistoryTab() {
  const [items, setItems] = useState<ReboundHistoryItem[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const load = () => void window.hts.reboundHistory().then(setItems);
  useEffect(() => {
    load();
    const off = window.hts.onReboundUnread(() => load()); // a new alert / a popup closed by hand
    return () => void off();
  }, []);
  const cur = items.find((x) => x.id === sel) || null;
  const pick = (it: ReboundHistoryItem) => {
    setSel(it.id);
    if (!it.read) void window.hts.reboundHistoryRead([it.id]).then(load);
  };
  const unread = items.filter((x) => !x.read).length;

  return (
    <div className="rb-hist">
      <div className="rb-hist-list">
        <div className="rb-hist-bar">
          <span className="muted">
            {items.length}건{unread ? ` · 미확인 ${unread}건` : ''}
          </span>
          <button type="button" className="mini" disabled={!unread} onClick={() => void window.hts.reboundHistoryRead(null).then(load)}>
            모두 확인
          </button>
          <button type="button" className="mini" disabled={!items.length} onClick={() => window.confirm('알림 히스토리를 모두 지울까요?') && void window.hts.reboundHistoryRemove(null).then(() => (setSel(null), load()))}>
            전체 삭제
          </button>
        </div>
        {!items.length && <p className="muted small rb-hist-empty">아직 알림이 없습니다. 감지되어 팝업이 뜬 알림이 여기에 쌓입니다 (최대 500건).</p>}
        {items.map((it) => (
          <div key={it.id} className={`rb-hist-row ${it.id === sel ? 'on' : ''} ${it.read ? '' : 'unread'}`} onClick={() => pick(it)}>
            <i title={it.read ? '확인함' : '미확인'} />
            <span className="rb-hist-when">{when(it.at)}</span>
            <b>{it.name}</b>
            <span className="muted">{it.code}</span>
            <span className="rb-hist-score">{it.res.score}점</span>
            {it.kakao && <span className={it.kakao === 'sent' ? 'rb-ok' : 'rb-warn'} title={it.kakao === 'sent' ? '카카오톡 전송됨' : `카카오톡 전송 실패: ${it.kakaoError || ''}`}>톡</span>}
          </div>
        ))}
      </div>
      <div className="rb-hist-detail">
        {!cur ? (
          <p className="muted">왼쪽 목록에서 알림을 고르면 내역이 표시됩니다.</p>
        ) : (
          <>
            <h3>
              {cur.name} <span className="muted">({cur.code})</span>
            </h3>
            <p>
              {new Date(cur.at + KST).toISOString().slice(0, 19).replace('T', ' ')} 감지 · 봉 {hhmm(cur.res.snap.t)} · {num(cur.price ?? cur.res.snap.close, 0)}원
              {cur.changePct === undefined ? '' : ` (${cur.changePct >= 0 ? '+' : ''}${num(cur.changePct, 2)}%)`} · 가점 {cur.res.score}점
            </p>
            <h4>필수 조건</h4>
            <ul>
              {COND_KEYS.map((k) => (
                <li key={k} className={cur.res.conds[k].ok ? 'ok' : 'no'}>
                  <b>{cur.res.conds[k].ok ? '✓' : '·'}</b> {COND_LABEL[k]} <span className="muted">{cur.res.conds[k].note}</span>
                </li>
              ))}
            </ul>
            <h4>가점</h4>
            <ul>
              {BONUS_KEYS.map((k) => (
                <li key={k} className={cur.res.bonus[k].ok ? 'ok' : 'no'}>
                  <b>{cur.res.bonus[k].ok ? '✓' : '·'}</b> {BONUS_LABEL[k]} <span className="muted">{cur.res.bonus[k].note}</span>
                </li>
              ))}
            </ul>
            <h4>지표</h4>
            <p className="rb-hist-ind">
              RSI {num(cur.res.snap.rsi)} / {num(cur.res.snap.rsiSignal)} · MACD {num(cur.res.snap.macd)} / {num(cur.res.snap.macdSignal)} · DI+ {num(cur.res.snap.diPlus)} / DI- {num(cur.res.snap.diMinus)} / ADX {num(cur.res.snap.adx)}
              <br />
              MA5 각도 {num(cur.res.snap.angle)}° · VWAP 대비 {num(cur.res.snap.vwapPct, 2)}% · MA5 {num(cur.res.snap.ma5, 0)} / MA20 {num(cur.res.snap.ma20, 0)} / MA60 {num(cur.res.snap.ma60, 0)}
            </p>
            {cur.kakao && <p className={cur.kakao === 'sent' ? 'rb-ok' : 'rb-warn'}>{cur.kakao === 'sent' ? '카카오톡으로 전송했습니다.' : `카카오톡 전송 실패: ${cur.kakaoError || ''}`}</p>}
            <div className="rb-hist-actions">
              <button type="button" onClick={() => (useStore.getState().setCodeAll(cur.code), useStore.getState().open('chart', { code: cur.code }))} title="이 종목을 열린 종목 창에 표시하고 차트 창을 앞으로">
                종목 열기
              </button>
              <button type="button" onClick={() => void window.hts.reboundHistoryPopup(cur.id)}>
                팝업으로 보기
              </button>
              <button type="button" onClick={() => void window.hts.reboundHistoryRemove([cur.id]).then(() => (setSel(null), load()))}>
                삭제
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

const day = (t: number) => (t ? new Date(t + KST).toISOString().slice(0, 16).replace('T', ' ') : '-');

export function KakaoTab({ enabled, onEnabled }: { enabled: boolean; onEnabled: (on: boolean) => void }) {
  const [st, setSt] = useState<KakaoState | null>(null);
  const [key, setKey] = useState('');
  const [secret, setSecret] = useState('');
  const [uri, setUri] = useState('');
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    void window.hts.reboundKakaoState().then((s) => (setSt(s), setUri(s.redirectUri)));
  }, []);
  if (!st) return null;
  const run = async (fn: () => Promise<{ ok: boolean; state: KakaoState; error?: string }>, okText: string) => {
    setBusy(true);
    setMsg('');
    try {
      const r = await fn();
      setSt(r.state);
      setMsg(r.ok ? okText : r.error || '실패했습니다.');
    } finally {
      setBusy(false);
    }
  };
  const sample = alertSummary({ code: '448900', name: '한국피아이엠', price: 38500, changePct: -2.41, res: null });

  return (
    <div className="rb-body rb-kakao">
      <section className="rb-top">
        <label className="check rb-onoff">
          <input type="checkbox" checked={enabled} onChange={(e) => onEnabled(e.target.checked)} /> <b>알림을 내 카카오톡으로 보내기</b>
        </label>
        <span className="muted">
          상태: <b>{!st.configured ? '키 미등록' : !st.loggedIn ? '로그인 필요' : '사용 가능'}</b>
          {st.lastSentAt ? ` · 마지막 전송 ${day(st.lastSentAt)}` : ''}
        </span>
        <span className="muted small">체크를 바꾼 뒤 아래 "저장"을 눌러야 적용됩니다. 팝업이 뜨는 알림마다 요약 1건을 "나와의 채팅"으로 보냅니다.</span>
      </section>

      <section>
        <h3>1. 카카오 앱 정보</h3>
        <div className="rb-kform">
          <label>
            REST API 키 {st.keyHint && <span className="muted">(저장됨: {st.keyHint})</span>}
            <input type="password" value={key} placeholder={st.configured ? '바꿀 때만 입력' : 'REST API 키'} onChange={(e) => setKey(e.target.value)} autoComplete="off" />
          </label>
          <label>
            클라이언트 시크릿 {st.hasSecret && <span className="muted">(저장됨)</span>}
            <input type="password" value={secret} placeholder="콘솔에서 켜져 있을 때만" onChange={(e) => setSecret(e.target.value)} autoComplete="off" />
          </label>
          <label className="wide">
            리다이렉트 URI (콘솔에 등록한 값과 똑같이)
            <input type="text" value={uri} onChange={(e) => setUri(e.target.value)} spellCheck={false} />
          </label>
        </div>
        <div className="rb-row">
          <button type="button" disabled={busy || (!key && !secret && uri === st.redirectUri)} onClick={() => void run(() => window.hts.reboundKakaoSave({ restKey: key, clientSecret: secret || undefined, redirectUri: uri }), '저장했습니다.').then(() => (setKey(''), setSecret('')))}>
            앱 정보 저장
          </button>
          {st.hasSecret && (
            <button type="button" disabled={busy} onClick={() => void run(() => window.hts.reboundKakaoSave({ clientSecret: '' }), '시크릿을 지웠습니다.')}>
              시크릿 지우기
            </button>
          )}
        </div>
      </section>

      <section>
        <h3>2. 카카오 로그인</h3>
        <p className="muted small">
          로그인 창에서 본인 카카오 계정으로 로그인하고 "카카오톡 메시지 전송"에 동의합니다. 로그인 정보는 이 PC의 Windows 계정으로 암호화해 저장하며, 약 2개월 동안 자동으로 갱신됩니다.
        </p>
        <div className="rb-row">
          <button type="button" className="primary" disabled={busy || !st.configured} onClick={() => void run(() => window.hts.reboundKakaoLogin(), '로그인했습니다.')}>
            {st.loggedIn ? '다시 로그인' : '카카오 로그인'}
          </button>
          <button type="button" disabled={busy || !st.loggedIn} onClick={() => void run(() => window.hts.reboundKakaoTest(), '테스트 메시지를 보냈습니다. 카카오톡 "나와의 채팅"을 확인하세요.')}>
            테스트 메시지 보내기
          </button>
          <button type="button" disabled={busy || !st.loggedIn} onClick={() => void run(() => window.hts.reboundKakaoLogout(), '로그아웃했습니다.')}>
            로그아웃
          </button>
          {st.loggedIn && <span className="muted small">로그인 유지: {day(st.refreshExpiresAt)} 까지 (쓸 때마다 연장)</span>}
        </div>
        {(msg || st.lastError) && <p className={msg && !/실패|오류|필요|닫았/.test(msg) ? 'rb-ok' : 'rb-warn'}>{msg || st.lastError}</p>}
      </section>

      <section>
        <h3>3. 봇 선별 결과도 카카오톡으로 받기</h3>
        <p className="muted small">
          봇 → 데이터 작업의 "선별 결과 메일"이 메일을 보낼 때, 선별 종목의 종목코드와 종목명만 카카오톡으로 같이 보냅니다. 메일은 봇(파이썬)이 보내므로 봇용 로그인 파일이 따로 필요합니다. 아래 버튼을 누르면 로그인 창이 한 번 더 뜨고, 파일 두 개를{' '}
          <b>{st.botDir}</b> 에 만듭니다.
        </p>
        <div className="rb-row">
          <button type="button" disabled={busy || !st.configured} onClick={() => void run(() => window.hts.reboundKakaoBotFiles(), '봇용 로그인 파일을 만들었습니다. 다음 선별 결과 메일부터 카카오톡으로도 옵니다.')}>
            {st.botFiles ? '봇용 로그인 파일 다시 만들기' : '봇용 로그인 파일 만들기'}
          </button>
          <span className={st.botFiles ? 'rb-ok' : 'muted'}>{st.botFiles ? '파일 있음 - 이 PC에서 도는 선별 작업은 카카오톡으로도 보냅니다' : '파일 없음 - 선별 결과는 메일로만 갑니다'}</span>
        </div>
        <p className="muted small">
          이 두 파일(kakao_config.json, kakao_token.json)은 암호화되지 않은 파일입니다. 라즈베리파이에서 작업을 돌린다면 두 파일을 파이의 ~/.kakao/ 폴더로 복사하세요. 2개월 넘게 한 번도 보내지 않으면 다시 만들어야 합니다.
        </p>
      </section>

      <section>
        <h3>보내는 내용 (예시)</h3>
        <pre className="rb-sample">{sample}</pre>
      </section>

      <section>
        <h3>카카오 디벨로퍼스 설정 순서</h3>
        <ol className="rb-guide">
          <li>developers.kakao.com 에 본인 카카오 계정으로 로그인 → 앱 → 앱 추가 (이름·회사명은 아무 값).</li>
          <li>앱 → 플랫폼 키 → REST API 키: 키 값을 복사해 위 "REST API 키"에 넣습니다. 같은 화면의 리다이렉트 URI에 위 주소(기본 http://localhost:5000/oauth)를 등록합니다.</li>
          <li>같은 화면에 클라이언트 시크릿이 "사용함"이면 그 값을 위에 넣고, "사용 안 함"이면 비워 둡니다.</li>
          <li>카카오 로그인 → 사용 설정: 상태를 ON 으로.</li>
          <li>카카오 로그인 → 동의항목 → 카카오톡 메시지 전송(talk_message): "선택 동의"로 설정.</li>
          <li>(선택) 메시지의 "종목 보기" 버튼을 쓰려면 앱 → 제품 링크 관리 → 웹 도메인에 https://m.stock.naver.com 을 등록합니다. 등록하지 않아도 메시지는 옵니다.</li>
          <li>여기서 "앱 정보 저장" → "카카오 로그인" → "테스트 메시지 보내기" 순서로 확인합니다.</li>
        </ol>
        <p className="muted small">본인에게 보내는 메시지라 비즈니스 인증이나 검수는 필요 없습니다. 카카오톡에서 "나와의 채팅" 알림이 꺼져 있으면 소리 없이 쌓이기만 합니다.</p>
      </section>
    </div>
  );
}
