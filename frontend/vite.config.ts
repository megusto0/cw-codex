import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

const backendPort = process.env.BACKEND_PORT ?? '8000';
const apiProxyTarget = process.env.VITE_API_BASE ?? `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': apiProxyTarget,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    maxWorkers: 1,
    minWorkers: 1,
  },
});
