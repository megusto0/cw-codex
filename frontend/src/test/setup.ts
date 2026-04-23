import '@testing-library/jest-dom/vitest';
import React from 'react';
import { vi } from 'vitest';

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// `recharts` relies on ResizeObserver even in jsdom.
// The lightweight mock is sufficient for static rendering tests.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).ResizeObserver = ResizeObserverMock;

vi.mock('recharts', () => {
  const PassThrough = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
  const Null = () => null;

  return {
    ResponsiveContainer: PassThrough,
    LineChart: PassThrough,
    BarChart: PassThrough,
    CartesianGrid: Null,
    Tooltip: Null,
    XAxis: Null,
    YAxis: Null,
    Legend: Null,
    Line: Null,
    Bar: Null,
    ReferenceLine: Null,
  };
});
