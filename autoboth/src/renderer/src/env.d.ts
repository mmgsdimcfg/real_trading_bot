import type { Api } from '../../preload/index';

declare global {
  interface Window {
    api: Api;
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
