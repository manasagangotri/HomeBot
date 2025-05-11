import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import styles from './Register.module.css';

function Register() {
  const [formData, setFormData] = useState({
    name: '',
    username: '',
    email: '',
    password: '',
  });
  const [error, setError] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [emailError, setEmailError] = useState('');
  const navigate = useNavigate();

  // Validate email format
  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // Validate password strength
  const validatePassword = (password) => {
    return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/.test(password);
  };

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });

    if (name === 'password') {
      setPasswordError(
        validatePassword(value)
          ? ''
          : 'Password must be at least 8 characters long and include uppercase, lowercase, a number, and a special character.'
      );
    }

    if (name === 'email') {
      setEmailError(validateEmail(value) ? '' : 'Enter a valid email address.');
    }
  };

  // Submit registration form
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!validateEmail(formData.email)) {
      setEmailError('Enter a valid email address.');
      return;
    }

    if (!validatePassword(formData.password)) {
      setPasswordError('Password must be strong.');
      return;
    }

    try {
      const res = await axios.post('http://localhost:5000/register', formData);
      console.log(res.data);
      navigate('/login');
    } catch (err) {
      setError(err.response?.data?.error || 'Registration failed');
    }
  };

  return (
    <div className={styles.container}>
      <h2>Register</h2>
      {error && <p className={styles.error}>{error}</p>}
      <form onSubmit={handleSubmit} className={styles.form}>
        <input type="text" name="name" placeholder="Full Name" value={formData.name} onChange={handleChange} required className={styles.input} />
        <input type="text" name="username" placeholder="Username" value={formData.username} onChange={handleChange} required className={styles.input} />
        <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} required className={styles.input} />
        {emailError && <p className={styles.validationError}>{emailError}</p>}
        <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} required className={styles.input} />
        {passwordError && <p className={styles.validationError}>{passwordError}</p>}
        <button type="submit" className={styles.button}>Register</button>
        <button type="button" className={styles.backButton} onClick={() => navigate('/')}>Back</button>
      </form>
    </div>
  );
}

export default Register;
