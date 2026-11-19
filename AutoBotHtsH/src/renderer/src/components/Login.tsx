// Update log
// - [2026-11-06] type=feat (Login: KIS app key + app secret, real / paper server, remember (encrypted with the Windows account), saved-key
//   login, demo mode. "조회 전용" is fixed - there is no account number field and no way to enable trading. The secret is sent to the
//   main process once and the field is cleared)

import { useState } from 'react';
import type { KisEnv } from '../../../shared/kis';
import { useStore } from '../store';

export function Login({ onClose }: { onClose?: () => void }) {
  const auth = useStore((s) => s.auth);
  const setAuth = useStore((s) => s.setAuth);
  const [env, setEnv] = useState<KisEnv>(auth?.env || 'real');
  const [key, setKey] = useState('');
  const [secret, setSecret] = useState('');
  const [remember, setRemember] = useState(auth?.remember ?? true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');

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
          <img src="./icon.png" alt="" /> HtsH 로그인
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
        {msg && <p className="login-msg">{msg}</p>}
        <p className="muted small">
          실전/모의 서버는 앱키를 발급받은 계좌 종류와 같아야 합니다. 같은 앱키를 다른 프로그램(자동매매 등)과 함께 쓰면 초당 호출 한도를 나눠 쓰게 되므로, 가능하면 HtsH용 앱키를 따로 쓰세요.
        </p>
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
