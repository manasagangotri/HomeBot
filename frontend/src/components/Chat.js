 

import { useState, useEffect, useRef } from "react";
import axios from "axios";
import styles from "./Chat.module.css";
import { useNavigate } from "react-router-dom";

function Chat() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [isVoiceInput, setIsVoiceInput] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentUtterance, setCurrentUtterance] = useState(null);
  const chatEndRef = useRef(null);
  const [feedbackSuggestions, setFeedbackSuggestions] = useState([]);
  const username = localStorage.getItem("username");
  const navigate = useNavigate();

  // Auto-scroll to the bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (overrideInput) => {
    const finalInput = overrideInput || (Array.isArray(input) ? input.join(", ") : input);
    if (!finalInput.trim()) return;

    setMessages((prevMessages) => [
      ...prevMessages,
      { text: finalInput, sender: "user" },
    ]);

    try {
      const res = await axios.post("http://localhost:5000/chat", {
        username: localStorage.getItem("username"),
        message: finalInput,
      });

      const botMessages = res.data.reply || "Sorry, I didn't get that.";
      const buttons = res.data.buttons || [];

      if (isVoiceInput && "speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(botMessages);
        utterance.lang = "en-US";
        window.speechSynthesis.speak(utterance);
        setCurrentUtterance(utterance);
      }

      setMessages((prevMessages) => [
        ...prevMessages,
        { text: botMessages, sender: "bot", buttons },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
    }

    setInput("");
    setIsVoiceInput(false);
  };

  const formatBotResponse = (text) => {
    const linkedText = text.replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer" style="color:blue;text-decoration:underline;">$1</a>'
    );
    return linkedText.replace(/\n/g, "<br />");
  };

  const startListening = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setSpeechSupported(false);
      alert("Speech Recognition not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);

    recognition.onresult = (event) => {
      const speechResult = event.results[0][0].transcript;
      setInput(speechResult);
      setIsVoiceInput(true);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error", event.error);
      setIsListening(false);
    };

    recognition.start();
  };

  const handlePause = () => {
    if (window.speechSynthesis.paused && currentUtterance) {
      window.speechSynthesis.resume();
      setIsPaused(false);
    } else if (window.speechSynthesis.speaking) {
      window.speechSynthesis.pause();
      setIsPaused(true);
    }
  };


  const handleFeedback = async (index, feedbackType, suggestionText) => {
    const botMessage = messages[index]?.text;
  
    try {
      await axios.post("http://localhost:5000/feedback", {
        username: localStorage.getItem("username"),
        message: botMessage,
        feedback: feedbackType,
        suggestion: suggestionText || "",
      });
      alert("Feedback submitted. Thanks!");
    } catch (err) {
      console.error("Feedback error:", err);
      alert("Failed to submit feedback.");
    }
  };
  

  return (
    <div className={styles.chatPage}>
      <div className={styles.chatContainer}>
        <button className={styles.backButton} onClick={() => navigate("/")}>
          ⬅ Back to Home
        </button>
        {username === "admin" && (
  <button
    className={styles.adminButton}
    onClick={() => navigate("/admin")}
  >
    🛠 Admin Dashboard
  </button>
)}

        {!speechSupported && (
          <div className={styles.warning}>
            ❗ Speech recognition not supported in your browser.
          </div>
        )}

        {isListening && (
          <div className={styles.listeningStatus}>🎙️ Listening...</div>
        )}

        <div className={styles.chatBox}>
          {messages.map((msg, index) => (
            <div
              key={index}
              className={
                msg.sender === "user"
                  ? styles.userMessage
                  : styles.botMessage
              }
            >
              {msg.sender === "bot" ? (
                <>
                  <div
                    dangerouslySetInnerHTML={{
                      __html: formatBotResponse(msg.text),
                    }}
                  />
            {/* Feedback section */}
            <div className={styles.feedbackSection}>
  <div className={styles.feedbackButtons}>
    <button
      className={styles.likeButton}
      onClick={() =>
        setFeedbackSuggestions((prev) => {
          const updated = [...prev];
          updated[index] = { ...(updated[index] || {}), type: "like" };
          return updated;
        })
      }
      style={{
        backgroundColor:
          feedbackSuggestions[index]?.type === "like" ? "lightgreen" : "",
      }}
    >
      👍
    </button>
    <button
      className={styles.dislikeButton}
      onClick={() =>
        setFeedbackSuggestions((prev) => {
          const updated = [...prev];
          updated[index] = { ...(updated[index] || {}), type: "dislike" };
          return updated;
        })
      }
      style={{
        backgroundColor:
          feedbackSuggestions[index]?.type === "dislike" ? "#f8bfbf" : "",
      }}
    >
      👎
    </button>
  </div>

{/* Suggestion Toggle Link */}
<button
  className={styles.suggestionLink}
  onClick={() => {
    const updated = [...feedbackSuggestions];
    updated[index] = {
      ...(updated[index] || {}),
      showSuggestion: !updated[index]?.showSuggestion,
    };
    setFeedbackSuggestions(updated);
  }}
>
  💬 Suggestion
</button>

{/* Conditionally Render Suggestion Textarea */}
{feedbackSuggestions[index]?.showSuggestion && (
  <>
    <textarea
      placeholder="Leave a suggestion"
      className={styles.feedbackInput}
      value={feedbackSuggestions[index]?.text || ""}
      onChange={(e) => {
        const updated = [...feedbackSuggestions];
        updated[index] = {
          ...(updated[index] || {}),
          text: e.target.value,
        };
        setFeedbackSuggestions(updated);
      }}
    />

    <button
      className={styles.submitFeedbackButton}
      onClick={() =>
        handleFeedback(
          index,
          feedbackSuggestions[index]?.type || "",
          feedbackSuggestions[index]?.text || ""
        )
      }
    >
      Submit Feedback
    </button>
  </>
)}
</div>


                </>
                
              ) : (
                msg.text
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
        
        <div className={styles.inputBox}>
          <input
            type="text"
            value={Array.isArray(input) ? input.join(", ") : input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message or use mic..."
          />
          <button
            onClick={startListening}
            className={styles.micButton}
            title="Click to speak"
          >
            🎤
          </button>
          <button onClick={() => handleSend()} className={styles.sendButton}>
            ➤
          </button>
          {currentUtterance && (
            <button
              onClick={handlePause}
              className={styles.pauseButton}
              title={isPaused ? "Resume Speech" : "Pause Speech"}
            >
              {isPaused ? "▶️" : "⏸️"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default Chat;
