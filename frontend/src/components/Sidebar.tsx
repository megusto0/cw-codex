import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Overview' },
  { to: '/cases', label: 'Case Explorer' },
  { to: '/retrieval', label: 'Similar Cases' },
  { to: '/prototypes', label: 'Prototype Memory' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/about', label: 'About' },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <span className="brand-kicker">Standalone Demo</span>
        <h1>RL Therapy Lab</h1>
        <p>Hopfield Postprandial Memory</p>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-note panel">
        <span className="eyebrow">Important</span>
        <p>This interface shows remembered historical patterns. It does not provide therapy or dosing advice.</p>
      </div>
    </aside>
  );
}

