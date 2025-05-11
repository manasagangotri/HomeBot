import { Link } from 'react-router-dom';
import styles from './Navbar.module.css';

function Navbar() {
  const role = localStorage.getItem("role"); // ✅ Define role here

  return (
    <nav className={styles.navbar}>
      <Link to="/" className={styles.logo}>Home Remedies</Link>
      <div>
        <Link to="/login" className={styles.link}>Login</Link>
        <Link to="/register" className={styles.link}>Register</Link>
        <Link to="/chat" className={styles.link}>Chat</Link>
        {role === "admin" && <Link to="/admin-dashboard">Admin Dashboard</Link>}

      </div>
    </nav>
  );
}

export default Navbar;
