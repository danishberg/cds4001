// react-app/src/components/ScenarioSimulator.tsx
import React, { useState } from 'react';
import { fetchPrediction } from '../services/api';
import '../App.css'; // Import CSS

// Interface for the result data (keep consistent)
interface SimulationResult {
  sleepHours: number;
  exerciseFreq: number;
  risk_score: number;
  classification: string;
  confidence: number;
  timestamp?: string; // Optional timestamp
}

// Define props for ScenarioSimulator, including the callback
interface ScenarioSimulatorProps {
  onNewResult: (result: SimulationResult) => void;
}

const ScenarioSimulator: React.FC<ScenarioSimulatorProps> = ({ onNewResult }) => {
  const [sleepHours, setSleepHours] = useState(7);
  const [exerciseFreq, setExerciseFreq] = useState(3);
  // State for the *latest* result to display immediately
  const [latestResult, setLatestResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async () => {
    setLoading(true);
    setError(null); // Clear previous errors
    setLatestResult(null); // Clear previous result display
    try {
      const data = await fetchPrediction({ sleepHours, exerciseFreq });
      const newResult: SimulationResult = {
        sleepHours,
        exerciseFreq,
        ...data,
        timestamp: new Date().toISOString(), // Add timestamp
      };
      setLatestResult(newResult); // Update local state for immediate display
      onNewResult(newResult); // Pass the result up to the Dashboard
    } catch (err) {
      console.error('Error running simulation:', err);
      setError('Simulation failed. Check connection or backend.');
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
    <div className="scenario-simulator">
      <h2>Scenario Simulator (What-If)</h2>
      <div className="simulator-controls">
        <div className="input-group">
          <label>
            Sleep Hours:
            <input
              type="range"
              min="4"
              max="12"
              step="0.5"
              value={sleepHours}
              onChange={(e) => setSleepHours(Number(e.target.value))}
              disabled={loading} // Disable during loading
            />
            <span>{sleepHours} hours</span>
          </label>
        </div>
        <div className="input-group">
          <label>
            Exercise Frequency (days/week):
            <input
              type="range"
              min="0"
              max="7"
              value={exerciseFreq}
              onChange={(e) => setExerciseFreq(Number(e.target.value))}
              disabled={loading} // Disable during loading
            />
            <span>{exerciseFreq} days</span>
          </label>
        </div>
        <button 
          onClick={runSimulation}
          disabled={loading}
          className="simulate-button"
        >
          {loading ? 'Running...' : 'Run Simulation'}
        </button>
      </div>

      {/* Display loading or error state */}
      {loading && <p>Calculating prediction...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* Display the latest result */}
      {latestResult && !loading && (
        <div className="results-container latest-result-display"> {/* Added class */}
            <h3>Prediction Result</h3>
            <div 
              className="risk-score" 
              style={{ 
                backgroundColor: getScoreColor(latestResult.risk_score)
              }}
              title={`Risk Score: ${latestResult.risk_score}`}
            >
              {latestResult.risk_score}
            </div>
            <div className="result-details">
              <p>Classification: <span style={{fontWeight: 'bold', color: getScoreColor(latestResult.risk_score)}}>{latestResult.classification}</span></p>
              <p>Confidence: {(latestResult.confidence * 100).toFixed(1)}%</p>
            </div>
        </div>
      )}

      {/* Note: The history chart is now moved to TrendAnalysis */}
    </div>
  );
};

export default ScenarioSimulator;
