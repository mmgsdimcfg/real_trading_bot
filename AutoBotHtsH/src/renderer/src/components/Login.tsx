// Update log
// - [2026-11-06] type=feat (Login: KIS app key + app secret, real / paper server, remember (encrypted with the Windows account), saved-key
//   login, demo mode. "조회 전용" is fixed - there is no account number field and no way to enable trading. The secret is sent to the
//   main process once and the field is cleared)
// - [2026-11-20] type=feat (글자 크기 in points 8 - 14 pt (default 9 pt) - the text size of the whole app, not a zoom)
// - [2026-11-20] type=feat (AutoBotHtsH: title "KIS 로그인 · 설정"; 화면 section with 글자 크기 (80 - 150 %, also Ctrl + = / - / 0) and a
//   link to the bot connection settings)
// - [2026-11-23] type=feat (Guest mode (user 2026-11-23): 게스트 접속 - host address + password instead of a KIS key (this PC then shows
//   the host's market data); 게스트 접속 허용 - on a PC logged in to KIS: relay on / off, port, password, this PC's addresses and
//   how many guests are connected)

import { useEffect, useState } from 'react';
import type { KisEnv } from '../../../shared/kis';
import { FONT_PT_DEFAULT, FONT_PTS, useStore } from '../store';
import type { GuestHostState } from '../../../preload';

/** [2026-11-20] 글자 크기 in points - one text size for the whole app (chart text 2px smaller, small marks 3px smaller) */
export function FontScaleRow() {
  const pt = useStore((s) => s.fontPt);
  const setPt = useStore((s) => s.setFontPt);
  return (
    <div className="font-row" title={`앱 전체 글자 크기 (포인트, 기본 ${FONT_PT_DEFAULT}pt). 단축키 Ctrl + = 크게, Ctrl + - 작게, Ctrl + 0 기본`}>
      <span>글자 크기</span>
      <span className="seg">
        {FONT_PTS.map((v) => (
          <button key={v} className={v === pt ? 'on' : ''} onClick={() => setPt(v)}>
            {v}pt
          </button>
        ))}
      </span>
    </div>
  );
}

/** [2026-11-23] this PC relays its KIS market data to guest PCs */
function GuestHostPanel() {
  const [st, setSt] = useState<GuestHostState | null>(null);
  const [port, setPort] = useState('');
  const [pass, setPass] = useState('');
  const [msg, setMsg] = useState('');
  const load = () => void window.hts.guestHostState().then((s) => (setSt(s), setPort((p) => p || String(s.port))));
  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  if (!st) return null;
  const save = async (enabled: boolean) => {
    const r = await window.hts.guestHostSave({ enabled, port: Number(port), password: pass || undefined });
    if (r.ok) (setSt(r.data), setPass(''), setMsg(r.data.error || (enabled ? '게스트 접속을 허용했습니다.' : '게스트 접속을 껐습니다.')));
    else setMsg(r.error);
  };
  return (
    <>
      <h3 className="login-sub">게스트 접속 허용 (이 PC가 시세 중계)</h3>
      <p className="muted small">인증키가 없는 다른 PC의 AutoBotHtsH가 이 PC를 통해 시세를 조회합니다. 본인 PC끼리만 쓰세요. 켜면 Windows 방화벽 허용 창이 뜰 수 있습니다.</p>
      <div className="guest-row">
        <label>
          포트
          <input value={port} onChange={(e) => setPort(e.target.value.replace(/\D/g, ''))} inputMode="numeric" />
        </label>
        <label>
          {st.hasPassword ? '암호 (바꿀 때만 입력)' : `암호 (${st.passMin}자 이상)`}
          <input type="password" value={pass} onChange={(e) => setPass(e.target.value)} autoComplete="new-password" />
        </label>
      </div>
      <div className="guest-row">
        <button className={st.enabled ? '' : 'primary'} onClick={() => void save(true)}>
          {st.enabled ? '설정 저장' : '게스트 접속 허용'}
        </button>
        {st.enabled && <button onClick={() => void save(false)}>끄기</button>}
      </div>
      <p className="muted small">
        {st.on ? `중계 중 · 게스트 ${st.clients}대 · 조회 ${st.calls.toLocaleString('ko-KR')}건` : st.enabled ? `중계 시작 실패${st.error ? ` - ${st.error}` : ''}` : '꺼짐'}
        {st.on && st.addresses.length > 0 && ` · 게스트 PC에 입력할 주소: ${st.addresses.map((a) => `${a}:${st.port}`).join(', ')}`}
      </p>
      {msg && <p className="login-msg">{msg}</p>}
    </>
  );
}

export function Login({ onClose }: { onClose?: () => void }) {
  const auth = useStore((s) => s.auth);
  const setAuth = useStore((s) => s.setAuth);
  const [env, setEnv] = useState<KisEnv>(auth?.env || 'real');
  const [key, setKey] = useState('');
  const [secret, setSecret] = useState('');
  const [remember, setRemember] = useState(auth?.remember ?? true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  const [gUrl, setGUrl] = useState(auth?.guest?.url || '');
  const [gPass, setGPass] = useState('');
  const [gRemember, setGRemember] = useState(true);
  const guest = async (useSaved: boolean) => {
    setBusy(true);
    setMsg('호스트에 접속 중…');
    const r = await window.hts.guestLogin({ url: gUrl, password: gPass, remember: gRemember, useSaved });
    setBusy(false);
    setGPass('');
    if (r.ok) {
      setAuth(r.data);
      setMsg('');
      onClose?.();
    } else setMsg(r.error);
  };

  const go = async (useSaved: boolean) => {
    setBusy(true);
    setMsg(useSaved ? '저장된 앱키로 접속 중…' : '접속 중… (접근 토큰 발급)');
    const r = await window.hts.login({ env, appKey: key, appSecret: secret, remember, useSaved });
    setBusy(false);
    setSecret('');
    if (r.ok) {
      setAuth(r.data);
      setMsg('');
      onClose?.();
    } else setMsg(r.error);
  };

  return (
    <div className="login-back">
      <div className="login">
        <h2>
          <img src="./icon.png" alt="" /> KIS 로그인 · 설정
        </h2>
        <p className="muted small">한국투자증권 KIS Developers에서 발급받은 앱키·앱시크릿으로 시세를 조회합니다.</p>
        <div className="seg">
          <button className={env === 'real' ? 'on' : ''} onClick={() => setEnv('real')}>
            실전투자 서버
          </button>
          <button className={env === 'paper' ? 'on' : ''} onClick={() => setEnv('paper')}>
            모의투자 서버
          </button>
        </div>
        {auth?.hasSaved && (
          <button className="primary wide" disabled={busy} onClick={() => void go(true)}>
            저장된 앱키로 접속 ({auth.keyHint})
          </button>
        )}
        <label>
          앱키 (App Key)
          <input value={key} onChange={(e) => setKey(e.target.value)} spellCheck={false} autoComplete="off" />
        </label>
        <label>
          앱시크릿 (App Secret)
          <input type="password" value={secret} onChange={(e) => setSecret(e.target.value)} spellCheck={false} autoComplete="off" />
        </label>
        <label className="check">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} /> 이 PC에 저장 (Windows 계정으로 암호화)
        </label>
        <label className="check locked" title="HtsH에는 주문 기능이 없습니다. 계좌번호를 받지 않고, 시세 조회 TR만 호출합니다.">
          <input type="checkbox" checked readOnly disabled /> 조회 전용 (매매 불가 - 변경할 수 없음)
        </label>
        <button className="primary wide" disabled={busy || !key || !secret} onClick={() => void go(false)}>
          접속
        </button>
        <button className="wide" disabled={busy} onClick={() => void window.hts.demo().then((a) => (setAuth(a), onClose?.()))}>
          데모 모드로 둘러보기 (가상 데이터)
        </button>
        <h3 className="login-sub">게스트 접속 (인증키 없는 PC)</h3>
        <p className="muted small">KIS에 로그인된 다른 PC의 AutoBotHtsH(호스트)를 통해 시세를 받습니다. 호스트 PC의 설정에서 "게스트 접속 허용"을 켠 뒤, 거기 나온 주소와 암호를 넣으세요.</p>
        {auth?.guest?.hasSaved && (
          <button className="primary wide" disabled={busy} onClick={() => void guest(true)}>
            저장된 호스트로 접속 ({auth.guest.url.replace(/^https?:\/\//, '')})
          </button>
        )}
        <div className="guest-row">
          <label>
            호스트 주소
            <input value={gUrl} onChange={(e) => setGUrl(e.target.value)} placeholder="192.168.0.10:47800" spellCheck={false} autoComplete="off" />
          </label>
          <label>
            암호
            <input type="password" value={gPass} onChange={(e) => setGPass(e.target.value)} autoComplete="off" />
          </label>
        </div>
        <label className="check">
          <input type="checkbox" checked={gRemember} onChange={(e) => setGRemember(e.target.checked)} /> 호스트와 암호를 이 PC에 저장 (Windows 계정으로 암호화)
        </label>
        <button className="wide" disabled={busy || !gUrl || !gPass} onClick={() => void guest(false)}>
          게스트로 접속
        </button>
        {msg && <p className="login-msg">{msg}</p>}
        <p className="muted small">
          실전/모의 서버는 앱키를 발급받은 계좌 종류와 같아야 합니다. 같은 앱키를 다른 프로그램(자동매매 등)과 함께 쓰면 초당 호출 한도를 나눠 쓰게 되므로, 가능하면 HtsH용 앱키를 따로 쓰세요.
        </p>
        {auth?.status === 'in' && auth.mode === 'kis' && <GuestHostPanel />}
        <h3 className="login-sub">화면</h3>
        <FontScaleRow />
        {auth?.status === 'in' && (
          <button className="wide" onClick={() => (useStore.getState().openBot('settings'), onClose?.())}>
            봇 설정·연결 열기 (SSH / 이 PC 실행 위치)
          </button>
        )}
        {onClose && auth?.status === 'in' && (
          <div className="actions">
            <button onClick={() => void window.hts.logout(false).then(setAuth)}>로그아웃</button>
            <button onClick={() => window.confirm('저장된 앱키와 토큰을 지울까요?') && void window.hts.logout(true).then(setAuth)}>저장된 앱키 지우기</button>
            <button onClick={onClose}>닫기</button>
          </div>
        )}
      </div>
    </div>
  );
}
