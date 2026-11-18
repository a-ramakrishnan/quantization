export function TraceReadout({ value, name, why, next, playing }: {
  value: string;
  name: string;
  why: string;
  next: string;
  playing: boolean;
}) {
  return (
    <dl className="operation-readout" aria-live={playing ? "off" : "polite"}>
      <div>
        <dt>What value is here?</dt>
        <dd><span className="font-mono text-theater-gold">{value}</span> <span className="text-theater-muted">· {name}</span></dd>
      </div>
      <div><dt>Why did it change?</dt><dd>{why}</dd></div>
      <div><dt>What happens next?</dt><dd>{next}</dd></div>
    </dl>
  );
}
