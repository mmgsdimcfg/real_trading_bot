// Update log
// - [2026-11-06] type=feat (Symbol search scoring: exact code > code prefix > name prefix > name contains > Korean initial consonants
//   (초성: "ㅅㅅㅈㅈ" matches 삼성전자, "ㅋㅋㅇ" matches 카카오); case-insensitive for Latin names (NAVER, KODEX))

const CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];

/** initial consonants of the Hangul syllables (other characters kept as they are, lower-cased) */
export function choseong(s: string): string {
  let out = '';
  for (const ch of s) {
    const c = ch.charCodeAt(0);
    if (c >= 0xac00 && c <= 0xd7a3) out += CHO[Math.floor((c - 0xac00) / 588)];
    else out += ch.toLowerCase();
  }
  return out;
}

const isAllCho = (s: string) => /^[ㄱ-ㅎ]+$/.test(s);

/** 0 = no match, higher = better */
export function matchScore(code: string, name: string, q: string): number {
  const s = q.trim();
  if (!s) return 0;
  const lower = s.toLowerCase();
  const nm = name.toLowerCase();
  if (code === s.toUpperCase()) return 100;
  if (/^[0-9A-Z]+$/i.test(s) && code.startsWith(s.toUpperCase())) return 80;
  if (nm === lower) return 90;
  if (nm.startsWith(lower)) return 70;
  if (nm.includes(lower)) return 50;
  if (isAllCho(s)) {
    const cs = choseong(name);
    if (cs.startsWith(s)) return 40;
    if (cs.includes(s)) return 30;
  }
  return 0;
}
