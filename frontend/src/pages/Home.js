import Navbar from '../components/Navbar';
import styles from './Home.module.css';

function Home() {
  return (
    <div className={styles.container}>
      <Navbar />
      <div className={styles.content}>
        <h1>Welcome to Home Remedies Chatbot</h1>
        <p>Get personalized home remedies for your health issues!</p>
      </div>
    </div>
  );
}

export default Home;
