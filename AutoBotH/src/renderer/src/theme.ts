/*
 * Update log
 * - [2026-11-07] type=feat - Theme choice (system / light / dark), kept per PC in localStorage and applied as <html data-theme> (no
 *   attribute = follow the system, see styles.css)
 */

export type ThemeChoice = 'system' | 'light' | 'dark';
const KEY = 'autoboth.theme';

export function loadTheme(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'light' || v === 'dark' ? v : 'system';
  } catch {
    return 'system';
  }
}

export function applyTheme(t: ThemeChoice): void {
  if (t === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage unavailable - applies for this session only */
  }
}

export const THEME_NEXT: Record<ThemeChoice, ThemeChoice> = { system: 'light', light: 'dark', dark: 'system' };
export const THEME_ICON: Record<ThemeChoice, string> = { system: '◐', light: '☀', dark: '☾' };
export const THEME_LABEL: Record<ThemeChoice, string> = { system: '시스템 설정 따름', light: '라이트', dark: '다크' };
