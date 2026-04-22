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

interface SeriesConfig {
  key: string;
  label: string;
  color: string;
  values: Array<number | null>;
}

interface WindowCurveChartProps {
  minutes: number[];
  series: SeriesConfig[];
  height?: number;
}

export default function WindowCurveChart({ minutes, series, height = 280 }: WindowCurveChartProps) {
  const chartData = minutes.map((minute, index) => {
    const record: Record<string, number | null> & { minute: number } = { minute };
    series.forEach((item) => {
      record[item.key] = item.values[index] ?? null;
    });
    return record;
  });

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 12, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(36, 56, 54, 0.09)" />
          <XAxis dataKey="minute" stroke="#57706d" tickLine={false} axisLine={false} />
          <YAxis stroke="#57706d" tickLine={false} axisLine={false} width={44} />
          <Tooltip
            contentStyle={{
              background: '#f9f8f3',
              border: '1px solid rgba(76, 104, 100, 0.18)',
              borderRadius: 14,
              boxShadow: '0 18px 40px rgba(45, 69, 66, 0.12)',
            }}
          />
          <ReferenceLine x={0} stroke="#a1b5b2" strokeDasharray="4 4" />
          <Legend />
          {series.map((item) => (
            <Line
              key={item.key}
              type="monotone"
              dataKey={item.key}
              name={item.label}
              dot={false}
              connectNulls={false}
              stroke={item.color}
              strokeWidth={2.4}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

