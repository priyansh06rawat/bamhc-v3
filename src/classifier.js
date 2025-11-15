import React, { useState, useEffect } from "react";
import { classifyText } from "./api";

const LabelColor = {
  hate: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
  offensive: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
  neither: "linear-gradient(135deg, #10b981 0%, #059669 100%)"
};

const LabelIcon = {
  hate: "⚠️",
  offensive: "⚡",
  neither: "✓"
};

// Diverse test examples
const testExamples = [
  { text: "I hope you get what you deserve", category: "Ambiguous" },
  { text: "You're absolutely worthless and nobody likes you", category: "Offensive" },
  { text: "I will find you and make you pay", category: "Threat" },
  { text: "Women belong in the kitchen", category: "Sexist" },
  { text: "All [ethnic group] are criminals", category: "Racist" },
  { text: "You are an amazing person, thank you!", category: "Positive" },
  { text: "This movie was terrible and boring", category: "Negative Opinion" },
  { text: "Go kill yourself", category: "Extreme Hate" },
  { text: "Your work is not up to standard", category: "Criticism" },
  { text: "I hate Mondays", category: "Casual Expression" },
  { text: "People like you shouldn't exist", category: "Dehumanizing" },
  { text: "What a beautiful day!", category: "Positive" },
  { text: "You stupid idiot", category: "Insult" },
  { text: "I love spending time with my family", category: "Wholesome" },
  { text: "This is the worst thing I've ever seen", category: "Strong Opinion" },
];

function Spinner() {
  return (
    <svg className="spinner" viewBox="0 0 50 50" aria-hidden="true">
      <circle 
        className="path" 
        cx="25" 
        cy="25" 
        r="20" 
        fill="none" 
        strokeWidth="4" 
      />
    </svg>
  );
}

export default function Classifier() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAllExamples, setShowAllExamples] = useState(false);
  const [history, setHistory] = useState(() => {
    try {
      const raw = localStorage.getItem("bamhc_history_v1");
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("bamhc_history_v1", JSON.stringify(history.slice(0, 20)));
    } catch {}
  }, [history]);

  async function onSubmit(e) {
    e && e.preventDefault();
    if (!text.trim()) return;
    
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await classifyText(text);
      const entry = {
        id: Date.now(),
        text,
        label: res.label,
        score: Number(res.score),
      };
      setResult(entry);
      setHistory(prev => [entry, ...prev].slice(0, 20));
    } catch (err) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  function addExample(t) {
    setText(t);
    // Scroll to textarea
    document.querySelector('.input-area')?.focus();
  }

  function clearHistory() {
    if (window.confirm("Clear all history?")) {
      setHistory([]);
    }
  }

  async function testAllExamples() {
    if (!window.confirm("This will test all 15 examples. Continue?")) return;
    
    setLoading(true);
    setError(null);
    
    const results = [];
    for (const example of testExamples) {
      try {
        const res = await classifyText(example.text);
        results.push({
          id: Date.now() + Math.random(),
          text: example.text,
          label: res.label,
          score: Number(res.score),
        });
        await new Promise(resolve => setTimeout(resolve, 500)); // Small delay
      } catch (err) {
        console.error('Error testing example:', err);
      }
    }
    
    setHistory(prev => [...results, ...prev].slice(0, 20));
    setLoading(false);
  }

  const charCount = text.length;
  const maxChars = 600;
  const charPercentage = (charCount / maxChars) * 100;
  const charColor = charPercentage > 90 ? '#ef4444' : charPercentage > 70 ? '#f59e0b' : '#94a3b8';

  const displayedExamples = showAllExamples ? testExamples : testExamples.slice(0, 6);

  return (
    <div className="ui-wrap">
      <div className="card">
        <h2>🛡️ Hate & Offensive Classifier</h2>
        <p className="muted">
          Advanced AI-powered content moderation. Paste text below to analyze for hate speech, 
          offensive language, and toxicity.
        </p>

        <form onSubmit={onSubmit} className="form">
          <textarea
            className="input-area"
            placeholder="Enter text to analyze for hate speech, offensive content, or toxicity..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            maxLength={maxChars}
            rows={5}
            aria-label="text to classify"
          />
          
          <div className="row">
            <div className="controls">
              <div className="char-count" style={{ color: charColor }}>
                {charCount}/{maxChars}
              </div>
              <button 
                className="primary" 
                type="submit" 
                disabled={loading || !text.trim()}
              >
                {loading ? (
                  <>
                    <Spinner /> 
                    Analyzing...
                  </>
                ) : (
                  <>
                    🔍 Classify
                  </>
                )}
              </button>
            </div>
          </div>
        </form>

        {/* Examples Section */}
        <div style={{ marginTop: '24px' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '16px'
          }}>
            <h3 style={{ fontSize: '18px', fontWeight: '700', margin: 0 }}>
              💡 Test Examples
            </h3>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                className="link-btn"
                onClick={testAllExamples}
                disabled={loading}
                style={{ 
                  fontSize: '13px',
                  padding: '6px 12px',
                  background: 'rgba(99, 102, 241, 0.1)',
                  border: '1px solid rgba(99, 102, 241, 0.3)',
                  borderRadius: '8px'
                }}
              >
                🧪 Test All
              </button>
              <button 
                className="link-btn"
                onClick={() => setShowAllExamples(!showAllExamples)}
              >
                {showAllExamples ? '📤 Show Less' : '📥 Show More'}
              </button>
            </div>
          </div>
          
          <div className="examples" style={{ gap: '8px' }}>
            {displayedExamples.map((example, idx) => (
              <button 
                key={idx}
                type="button" 
                className="chip" 
                onClick={() => addExample(example.text)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  padding: '10px 14px',
                  textAlign: 'left',
                  minWidth: '200px'
                }}
              >
                <span style={{ 
                  fontSize: '11px', 
                  opacity: 0.6,
                  textTransform: 'uppercase',
                  fontWeight: '600',
                  letterSpacing: '0.5px'
                }}>
                  {example.category}
                </span>
                <span style={{ 
                  fontSize: '13px', 
                  marginTop: '4px',
                  lineHeight: '1.4'
                }}>
                  {example.text.length > 50 ? example.text.slice(0, 50) + '...' : example.text}
                </span>
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="error">
            ❌ <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result">
            <div className="result-row">
              <div 
                className="label-pill" 
                style={{ background: LabelColor[result.label] }}
              >
                {LabelIcon[result.label]} {result.label.toUpperCase()}
              </div>
              <div className="score">
                {(result.score * 100).toFixed(1)}%
              </div>
            </div>
            <div className="score-bar">
              <div 
                className="score-inner" 
                style={{ 
                  width: `${result.score * 100}%`, 
                  background: LabelColor[result.label] 
                }} 
              />
            </div>
            <div className="result-text">
              "{result.text.length > 150 ? result.text.slice(0, 150) + '...' : result.text}"
            </div>
          </div>
        )}

        <div className="history-header">
          <h3>📜 Recent Classifications</h3>
          {history.length > 0 && (
            <button className="link-btn" onClick={clearHistory}>
              🗑️ Clear All
            </button>
          )}
        </div>

        <div className="history">
          {history.length === 0 && (
            <div className="muted" style={{ textAlign: 'center', padding: '20px' }}>
              No recent predictions yet. Try classifying some text above!
            </div>
          )}
          {history.map(h => (
            <div key={h.id} className="history-item">
              <div className="history-left">
                <div 
                  className="history-label" 
                  style={{ 
                    background: LabelColor[h.label],
                    color: 'white'
                  }}
                >
                  {LabelIcon[h.label]} {h.label}
                </div>
                <div className="history-score">
                  {(h.score * 100).toFixed(0)}%
                </div>
              </div>
              <div className="history-text">
                {h.text.length > 120 ? h.text.slice(0, 120) + '...' : h.text}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}