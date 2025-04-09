// react-app/src/components/RiskScore.tsx
import React, { useState, useEffect } from 'react';
import { fetchPrediction } from '../services/api';
import '../App.css'; // Import CSS for styling

const RiskScore: React.FC = () => {
  const [risk, setRisk] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPrediction({ sleepHours: 7, exerciseFreq: 3 })
      .then(data => {
        setRisk(data);
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching prediction:", error);
        setLoading(false);
      });
  }, []);

  const getScoreColor = (score: number) => {
    if (score < 40) return '#4CAF50'; // Green
    if (score < 70) return '#FFC107'; // Amber
    return '#F44336'; // Red
  };

  return (
    <div className="risk-score-container">
      <h2>Risk Score & Classification</h2>
      {loading ? (
        <p>Loading...</p>
      ) : risk ? (
        <div className="risk-details">
          <div 
            className="risk-gauge" 
            style={{
              borderColor: getScoreColor(risk.risk_score)
            }}
          >
            <span className="risk-value" style={{ color: getScoreColor(risk.risk_score) }}>
              {risk.risk_score}
            </span>
          </div>
          <p className="risk-classification">
            Classification: <span style={{ fontWeight: 'bold', color: getScoreColor(risk.risk_score) }}>{risk.classification}</span>
          </p>
          <p className="risk-confidence">Confidence: {(risk.confidence * 100).toFixed(1)}%</p>
        </div>
      ) : (
        <p>Error loading data.</p>
      )}
    </div>
  );
};

export default RiskScore;
