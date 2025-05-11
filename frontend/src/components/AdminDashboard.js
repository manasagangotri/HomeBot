
import { useEffect, useState } from "react";
import axios from "axios";
import styles from "./AdminDashboard.module.css";
import { useNavigate } from "react-router-dom";

function AdminDashboard() {
  const [feedbackList, setFeedbackList] = useState([]);
  const [feedbackStats, setFeedbackStats] = useState({
    total: 0,
    likes: 0,
    dislikes: 0,
    suggestions: 0,
  });
  const navigate = useNavigate();

  // Fetch Feedback and Perform Analytics
  useEffect(() => {
    const fetchFeedback = async () => {
      

      try {
        const res = await axios.get("http://localhost:5000/admin/feedback");
        setFeedbackList(res.data);

        // Analyzing feedback data
        const stats = {
          total: res.data.length,
          likes: res.data.filter((fb) => fb.feedback_type === "like").length,
          dislikes: res.data.filter((fb) => fb.feedback_type === "dislike").length,
          suggestions: res.data.filter((fb) => fb.suggestion).length,
        };
        setFeedbackStats(stats);
      } catch (err) {
        console.error("Failed to fetch feedback:", err);
      }
    };

    fetchFeedback();
  });

  // Function to Export Feedback as CSV
  const exportToCSV = () => {
    const headers = ["User", "Message", "Feedback", "Suggestion", "Date"];
    const rows = feedbackList.map((fb) => [
      fb.username,
      fb.bot_reply,
      fb.feedback_type,
      fb.suggestion,
      new Date(fb.created_at).toLocaleString(),
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      headers.join(",") +
      "\n" +
      rows.map((row) => row.join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "feedback_data.csv");
    document.body.appendChild(link);
    link.click();
  };

  return (
    <div className={styles.dashboard}>
      <button className={styles.backButton} onClick={() => navigate("/")}>
        ⬅ Back to Home
      </button>
      <h2>Admin Feedback Dashboard</h2>

      {/* Display Feedback Analytics */}
      <div className={styles.analytics}>
        <h3>Feedback Overview</h3>
        <p>Total Feedback: {feedbackStats.total}</p>
        <p>Likes: {feedbackStats.likes}</p>
        <p>Dislikes: {feedbackStats.dislikes}</p>
        <p>Suggestions: {feedbackStats.suggestions}</p>
      </div>

      {/* Feedback Table */}
      <table className={styles.feedbackTable}>
        <thead>
          <tr>
            <th>User</th>
            <th>Message</th>
            <th>Feedback</th>
            <th>Suggestion</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {feedbackList.map((fb) => (
            <tr key={fb.id}>
              <td>{fb.username}</td>
              <td>{fb.bot_reply}</td>
              <td>{fb.feedback_type}</td>
              <td>{fb.suggestion}</td>
              <td>{new Date(fb.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Export CSV Button */}
      <button className={styles.exportButton} onClick={exportToCSV}>
        📥 Export Feedback as CSV
      </button>
    </div>
  );
}

export default AdminDashboard;
