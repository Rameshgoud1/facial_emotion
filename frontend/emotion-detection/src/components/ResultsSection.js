import React from 'react';

function ResultsSection({ emotions }) {
  return (
    <div className="emotions-results">
      <h2>Detected Emotions:</h2>
      {emotions.map((emotion, index) => (
        <div key={index} className="emotion-card">
          <h3>Face {index + 1}</h3>
          <p>Emotion: {emotion.emotion}</p>
          <p>Confidence: {(emotion.confidence * 100).toFixed(2)}%</p>
        </div>
      ))}
    </div>
  );
}

export default ResultsSection;