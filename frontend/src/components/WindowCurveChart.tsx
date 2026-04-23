import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CHART_COLORS } from '../presentation';

interface SeriesConfig {
  key: string;
  label: string;
  values: Array<number | null>;
}

interface WindowCurveChartProps {
  minutes: number[];
  series: SeriesConfig[];
  height?: number;
}

const LINE_COLORS = [
  CHART_COLORS.accent,
  CHART_COLORS.green,
  CHART_COLORS.blue,
  CHART_COLORS.purple,
];

export default function WindowCurveChart({ minutes, series, height = 260 }: WindowCurveChartProps) {
  const chartData = minutes.map((minute, index) => {
    const record: Record<string, number | null> & { minute: number } = { minute };
    series.forEach((item) => {
      record[item.key] = item.values[index] ?? null;
    });
    return record;
  });

  return (
    <div className="chart-shell" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
          <XAxis dataKey="minute" stroke={CHART_COLORS.textSecondary} tickLine={false} axisLine={false} />
          <YAxis stroke={CHART_COLORS.textSecondary} tickLine={false} axisLine={false} width={42} />
          <Tooltip
            formatter={(value: number | string | Array<number | string | null> | null | undefined) =>
              typeof value === 'number' ? [`${value.toFixed(1)} мг/дл`, 'CGM'] : value
            }
            labelFormatter={(value) => `${value} мин`}
            contentStyle={{
              background: CHART_COLORS.tooltipBackground,
              border: `1px solid ${CHART_COLORS.tooltipBorder}`,
              borderRadius: 12,
              color: '#e8e8e8',
            }}
          />
          <ReferenceLine x={0} stroke={CHART_COLORS.textSecondary} strokeDasharray="4 4" />
          <Legend />
          {series.map((item, index) => (
            <Line
              key={item.key}
              type="monotone"
              dataKey={item.key}
              name={item.label}
              dot={false}
              connectNulls={false}
              stroke={LINE_COLORS[index % LINE_COLORS.length]}
              strokeWidth={2.2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
