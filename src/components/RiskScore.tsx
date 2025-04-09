// react-app/src/components/RiskScore.tsx
import React, { useState, useEffect } from 'react';
import { fetchPrediction } from '../services/api';

const RiskScore: React.FC = () => {
  const [risk, setRisk] = useState<any>(null);

  useEffect(() => {
    fetchPrediction({ sleepHours: 7, exerciseFreq: 3 }).then(data => {
      setRisk(data);
    });
  }, []);

  return (
    <div>
      <h2>Risk Score & Classification</h2>
      {risk ? (
        <div>
          <p>Risk Score: {risk.risk_score}</p>
          <p>Classification: {risk.classification}</p>
          <p>Confidence: {risk.confidence}</p>
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
};

export default RiskScore;
