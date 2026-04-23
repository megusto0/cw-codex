interface SomCell {
  cell_index: number;
  row: number;
  col: number;
  count: number;
  dominant_label?: string | null;
  dominant_label_display?: string;
  purity?: number;
}

interface SomGridChartProps {
  gridHeight: number;
  gridWidth: number;
  activeCell: number;
  cells: SomCell[];
}

export default function SomGridChart({ gridHeight, gridWidth, activeCell, cells }: SomGridChartProps) {
  const indexed = new Map(cells.map((cell) => [cell.cell_index, cell]));

  return (
    <div
      className="som-grid"
      style={{ gridTemplateColumns: `repeat(${gridWidth}, minmax(0, 1fr))` }}
      role="img"
      aria-label="Карта Кохонена"
    >
      {Array.from({ length: gridHeight * gridWidth }, (_, index) => {
        const cell = indexed.get(index);
        const purity = cell?.purity ?? 0;
        return (
          <div
            key={index}
            className={`som-cell${index === activeCell ? ' active' : ''}${cell && cell.count > 0 ? ' filled' : ''}`}
          >
            <strong>{cell?.count ?? 0}</strong>
            <span>{cell?.dominant_label_display ?? 'Пусто'}</span>
            <small>{cell ? `purity ${purity.toFixed(2)}` : '—'}</small>
          </div>
        );
      })}
    </div>
  );
}
