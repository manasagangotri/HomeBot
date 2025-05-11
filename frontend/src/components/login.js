import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import styles from "./Login.module.css"; // Import scoped CSS module

function Login() {
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://localhost:5000/login", formData);
      setMessage(res.data.message);
      if (res.status === 200) {
        localStorage.setItem("username", formData.username); 
        localStorage.setItem("role", res.data.role);   
        navigate("/chat"); // Redirect to chat on successful login
      }
    } catch (error) {
      setMessage(error.response?.data?.error || "Login failed");
    }
  };


  


  
  return (
    <div className={styles["login-container"]}>
      <form onSubmit={handleSubmit} className={styles["login-form"]}>
        <h2>Login</h2>
        <input
          type="text"
          name="username"
          className={styles["login-input"]}
          onChange={handleChange}
          placeholder="Username"
          required
        />
        <input
          type="password"
          name="password"
          className={styles["login-input"]}
          onChange={handleChange}
          placeholder="Password"
          required
        />
        <button type="submit" className={styles["login-button"]}>
          Login
        </button>
        <p className={styles["login-message"]}>{message}</p>
      </form>
    </div>
  );
}

export default Login;

