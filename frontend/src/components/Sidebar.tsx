import { NavLink } from 'react-router-dom';
import { useModelPreference } from '../model-context';

const navItems = [
  { to: '/', label: 'Обзор' },
  { to: '/explore', label: 'Поиск случаев' },
  { to: '/evaluation', label: 'Сравнение моделей' },
  { to: '/about', label: 'Методология' },
];

export default function Sidebar() {
  const { linkWithModel } = useModelPreference();

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <span className="brand-kicker">Ретроспективный исследовательский прототип</span>
        <h1>Postprandial Retrieval Lab</h1>
        <p>Сравнение нейронных сетей в задаче поиска сходных постпрандиальных CGM-окон.</p>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={linkWithModel(item.to)}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-note">
        <strong>Ограничение</strong>
        <p>
          Система не является медицинским изделием, не формирует клинические рекомендации и не
          предназначена для выбора дозы инсулина.
        </p>
      </div>
    </aside>
  );
}
