import React, { useState } from 'react';
import { fetchPrediction } from '../services/api';
import '../App.css';

interface UserInputProps {
  // Optional: Callback if dashboard needs to know about the prediction
  onNewPrediction?: (result: any) => void; 
}

const UserInputSurvey: React.FC<UserInputProps> = ({ onNewPrediction }) => {
  const [sleepHours, setSleepHours] = useState<number | string>(''); // Use string to allow empty input
  const [exerciseFreq, setExerciseFreq] = useState<number | string>('');
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault(); // Prevent default form submission
    setLoading(true);
    setError(null);
    setPrediction(null);

    // Basic validation
    const sleepNum = Number(sleepHours);
    const exerciseNum = Number(exerciseFreq);

    if (isNaN(sleepNum) || isNaN(exerciseNum) || sleepNum <= 0 || exerciseNum < 0) {
      setError("Please enter valid positive numbers for sleep and exercise.");
      setLoading(false);
      return;
    }
    
    // Range checks (optional but recommended)
    if (sleepNum < 4 || sleepNum > 12) {
        console.warn("Sleep hours outside typical range (4-12).");
    }
    if (exerciseNum > 7) {
        console.warn("Exercise frequency outside typical range (0-7).");
    }

    try {
      const result = await fetchPrediction({ sleepHours: sleepNum, exerciseFreq: exerciseNum });
      setPrediction(result);
      if (onNewPrediction) {
        onNewPrediction(result);
      }
    } catch (err) {
      console.error("Error getting prediction:", err);
      setError("Failed to get prediction. Check connection or backend.");
    }
    setLoading(false);
  };
  
  // Function to determine color based on risk score
  const getScoreColor = (score: number) => {
    if (score < 40) return '#4CAF50'; // Green
    if (score < 70) return '#FFC107'; // Amber
    return '#F44336'; // Red
  };

  return (
    <div className="user-input-survey-container">
      <h2>Get Your Health Risk Prediction</h2>
      <form onSubmit={handleSubmit} className="user-input-form">
        <div className="input-group">
          <label htmlFor="sleepHoursInput">Average Sleep Hours per Night:</label>
          <input
            id="sleepHoursInput"
            type="number"
            step="0.5" // Allow half hours
            min="1" 
            max="20" // Generous max
            value={sleepHours}
            onChange={(e) => setSleepHours(e.target.value)}
            required
            placeholder="e.g., 7.5"
            disabled={loading}
          />
        </div>
        <div className="input-group">
          <label htmlFor="exerciseFreqInput">Exercise Frequency (days per week):</label>
          <input
            id="exerciseFreqInput"
            type="number"
            step="1"
            min="0"
            max="7"
            value={exerciseFreq}
            onChange={(e) => setExerciseFreq(e.target.value)}
            required
            placeholder="e.g., 3"
            disabled={loading}
          />
        </div>
        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Calculating...' : 'Get Prediction'}
        </button>
      </form>

      {/* Display loading or error state */}
      {loading && <p>Calculating prediction...</p>}
      {error && <p style={{ color: 'red' }} className="error-message">{error}</p>}

      {/* Display the prediction result */}
      {prediction && !loading && (
        <div className="prediction-result-display">
          <h3>Your Predicted Risk</h3>
           <div 
              className="risk-gauge" 
              style={{ 
                borderColor: getScoreColor(prediction.risk_score),
                margin: '15px auto' // Center gauge
              }}
              title={`Risk Score: ${prediction.risk_score}`}
            >
                 <span className="risk-value" style={{ color: getScoreColor(prediction.risk_score) }}>
                    {prediction.risk_score}
                 </span>
            </div>
          <div className="result-details" style={{textAlign: 'center'}}>
             <p className="risk-classification">
                Classification: <span style={{fontWeight: 'bold', color: getScoreColor(prediction.risk_score)}}>{prediction.classification}</span>
             </p>
             <p className="risk-confidence">Model Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserInputSurvey; 