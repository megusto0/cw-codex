import { Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import AboutPage from './pages/AboutPage';
import CaseExplorerPage from './pages/CaseExplorerPage';
import DashboardPage from './pages/DashboardPage';
import EvaluationPage from './pages/EvaluationPage';
import PrototypesPage from './pages/PrototypesPage';
import RetrievalPage from './pages/RetrievalPage';

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/cases" element={<CaseExplorerPage />} />
          <Route path="/retrieval" element={<RetrievalPage />} />
          <Route path="/prototypes" element={<PrototypesPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  );
}
