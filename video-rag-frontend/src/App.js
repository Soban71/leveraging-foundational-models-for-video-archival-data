import { useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [clips, setClips] = useState([]);
  const [answer, setAnswer] = useState("");
  const [loadingClips, setLoadingClips] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async (event) => {
    event.preventDefault();

    if (!question.trim()) return;

    setLoadingClips(true);
    setGenerating(false);
    setClips([]);
    setAnswer("");
    setError("");

    try {
      // Step 1: Retrieve and show the five video clips quickly
      const retrievalResponse = await fetch(`${API_BASE}/api/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });

      if (!retrievalResponse.ok) {
        throw new Error("Could not retrieve video clips.");
      }

      const retrievalData = await retrievalResponse.json();
      setClips(retrievalData.clips || []);
      setLoadingClips(false);

      // Step 2: Generate the slower Ollama answer in the background
      setGenerating(true);

      const answerResponse = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });

      if (!answerResponse.ok) {
        throw new Error("Clips were retrieved, but Ollama could not generate an answer.");
      }

      const answerData = await answerResponse.json();
      setAnswer(answerData.answer || "No answer was generated.");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingClips(false);
      setGenerating(false);
    }
  };

  return (
    <div className="app">
      <header className="top-nav">
  <div className="brand">
    <div className="brand-mark">◈</div>
    <div>
      <span className="brand-name">ArchiveLens</span>
      <span className="brand-label">MULTIMODAL VIDEO RAG</span>
    </div>
  </div>

  <div className="nav-status">
    <span className="status-dot"></span>
    Local AI system active
  </div>
</header>
      <header className="hero">
        <p className="eyebrow">VIDEO ARCHIVE RAG</p>
        <h1>Search your video archive</h1>
        <p className="subtitle">
          Retrieval uses 70% visual information and 30% subtitle/Whisper text.
        </p>

        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Example: Children playing football"
          />
          <button type="submit" disabled={loadingClips || generating}>
            {loadingClips ? "Searching..." : "Search archive"}
          </button>
        </form>
      </header>

      {error && <div className="error-box">{error}</div>}

      {loadingClips && (
        <section className="answer-card">
          <h2>Finding relevant clips...</h2>
          <p>Please wait a moment.</p>
        </section>
      )}

      {clips.length > 0 && (
        <section className="clips-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">RETRIEVAL RESULTS</p>
              <h2>Relevant video clips</h2>
            </div>
            <span>{clips.length} clips retrieved</span>
          </div>

          <div className="clips-grid">
            {clips.map((clip) => (
              <article className="clip-card" key={clip.clip_id}>
                <div className="video-wrap">
                  <span className="rank">#{clip.rank}</span>
                  <span className="match">
                    {Math.round(clip.score * 100)}% match
                  </span>

                  <video
                    controls
                    preload="metadata"
                    className="clip-video"
                  >
                    <source
                      src={`${API_BASE}${clip.clip_url}`}
                      type="video/mp4"
                    />
                    Your browser does not support video playback.
                  </video>
                </div>

                <div className="clip-info">
                  <h3>{clip.clip_id}</h3>
                  <div className="scores">
                    <span>Visual: {clip.visual_score}</span>
                    <span>Text: {clip.text_score}</span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      {generating && (
        <section className="answer-card loading-answer">
          <p className="eyebrow">GENERATED ANSWER</p>
          <h2>Ollama is generating a grounded answer...</h2>
          <div className="loading-bar">
            <div className="loading-progress"></div>
          </div>
          <p>This can take 2–4 minutes on your local computer.</p>
        </section>
      )}

      {answer && !generating && (
        <section className="answer-card">
          <p className="eyebrow">GENERATED ANSWER</p>
          <h2>Archive response</h2>
          <p className="answer-text">{answer}</p>
        </section>
      )}

      <section className="about-section">
  <div className="about-tag">ABOUT THE PROJECT</div>

  <div className="about-content">
    <h2>Making video archives easier to explore.</h2>

    <div className="about-text">
      <p>
        ArchiveLens is a multimodal Video Retrieval-Augmented Generation
        system for searching a video archive using natural-language questions.
      </p>

      <p>
        It retrieves the five most relevant video clips by combining visual
        frame information (70%) with subtitles or Whisper-generated text (30%).
      </p>

      <p>
        The retrieved clips, transcripts and manually created descriptions are
        then provided to a local Llama model to generate a grounded answer.
      </p>

      <p>
        This helps users discover meaningful video evidence quickly, rather
        than manually watching a large archive.
      </p>
    </div>
  </div>
</section>

<footer className="footer">
  <div>
    <span className="footer-brand">ArchiveLens</span>
    <span className="footer-copy">
      © {new Date().getFullYear()} Video Archive Retrieval Project
    </span>
  </div>

  <div className="footer-tech">
    OpenCLIP <span>•</span> Whisper <span>•</span> ChromaDB <span>•</span> Ollama
  </div>
</footer>
    </div>
  );
}

export default App;