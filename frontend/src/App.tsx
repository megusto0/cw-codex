import { Navigate, Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import { ModelProvider } from './model-context';
import AboutPage from './pages/AboutPage';
import DashboardPage from './pages/DashboardPage';
import EvaluationPage from './pages/EvaluationPage';
import ExplorePage from './pages/ExplorePage';

export default function App() {
  return (
    <ModelProvider>
      <div className="app-shell">
        <Sidebar />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/explore" element={<ExplorePage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </ModelProvider>
  );
}
