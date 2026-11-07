import { useStore } from '../store';

export function SymbolPicker({ allowAll = true }: { allowAll?: boolean }) {
  const snap = useStore((s) => s.snapshot);
  const selected = useStore((s) => s.selected);
  const select = useStore((s) => s.select);
  return (
    <select className="picker" value={selected || ''} onChange={(e) => select(e.target.value || null)}>
      {allowAll && <option value="">전체 종목 (깔때기)</option>}
      {!allowAll && <option value="">종목 선택…</option>}
      {snap?.symbols.map((s) => (
        <option key={s.code} value={s.code}>
          {s.position ? '● ' : ''}
          {s.code} {s.name}
        </option>
      ))}
    </select>
  );
}
