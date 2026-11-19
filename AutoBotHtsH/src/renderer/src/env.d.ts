// Update log
// - [2026-11-06] type=feat (window globals from the preload bridge and the shared menu script)
import type { Hts } from '../../preload/index';

declare global {
  interface Window {
    hts: Hts;
    winCtl: {
      minimize: () => void;
      toggleMaximize: () => void;
      close: () => void;
      isMaximized: () => Promise<boolean>;
      onMaximized: (cb: (m: boolean) => void) => () => void;
    };
    AppMenu?: { init: (opts: Record<string, unknown>) => void };
  }
}

export {};
