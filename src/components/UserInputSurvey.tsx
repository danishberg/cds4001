import React, { useState } from 'react';
import { fetchPrediction } from '../services/api';
import '../App.css';

// Keep SimulationResult structure consistent
interface PredictionResult {
  risk_score: number;
  classification: string;
  confidence: number;
}

interface UserInputProps {
  // Callback to pass the full user input and prediction up
  onPredictionComplete?: (input: UserData, result: PredictionResult) => void; 
}

// Interface for user input data
interface UserData {
    sleepHours: number | string;
    exerciseFreq: number | string;
    age: number | string;
    gender: string;
    stressLevel: number | string;
}

// Define consistent color mapping based on classification
const classificationColors: { [key: string]: string } = {
    Low: '#4CAF50',      // Green
    Moderate: '#FFC107', // Amber
    High: '#F44336',      // Red
    Unknown: '#ccc'       // Grey
};

const UserInputSurvey: React.FC<UserInputProps> = ({ onPredictionComplete }) => {
  const [userInput, setUserInput] = useState<UserData>({
      sleepHours: '', 
      exerciseFreq: '',
      age: '',
      gender: 'Male', // Default gender
      stressLevel: '5' // Default stress level (assuming 1-10)
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = event.target;
    setUserInput(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault(); 
    setLoading(true);
    setError(null);
    setPrediction(null);

    // Basic validation
    const sleepNum = Number(userInput.sleepHours);
    const exerciseNum = Number(userInput.exerciseFreq);
    const ageNum = Number(userInput.age);
    const stressNum = Number(userInput.stressLevel);

    if (isNaN(sleepNum) || isNaN(exerciseNum) || isNaN(ageNum) || isNaN(stressNum) || 
        sleepNum <= 0 || exerciseNum < 0 || ageNum <= 0 || stressNum < 0 || stressNum > 10) {
      setError("Please enter valid positive numbers for all fields (Stress: 0-10).");
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
    if (ageNum < 18 || ageNum > 100) {
        console.warn("Age outside typical range (18-100).");
    }

    try {
      // Send all fields required by the current model
      const params = {
            sleepHours: sleepNum,
            exerciseFreq: exerciseNum,
            age: ageNum,
            gender: userInput.gender,
            stressLevel: stressNum
      };
      const result = await fetchPrediction(params); 
      setPrediction(result);
      if (onPredictionComplete) {
        // Pass the full input and the result up
        onPredictionComplete(params, result);
      }
    } catch (err) {
      console.error("Error getting prediction:", err);
      setError("Failed to get prediction. Check connection or backend.");
    }
    setLoading(false);
  };
  
  // Use the shared color map
  const getScoreColor = (classification: string) => {
    return classificationColors[classification] || classificationColors.Unknown;
  };

  return (
    <div className="user-input-survey-container card-content">
      <h2>Get Your Health Risk Prediction</h2>
      <p className="component-description">Enter your typical habits and demographics to get a personalized risk assessment based on our model.</p>
      <form onSubmit={handleSubmit} className="user-input-form">
        {/* Sleep Hours */}
        <div className="input-group">
          <label htmlFor="sleepHoursInput">Average Sleep Hours per Night:</label>
          <input
            id="sleepHoursInput"
            name="sleepHours"
            type="number"
            step="0.5" 
            min="1" 
            max="20" 
            value={userInput.sleepHours}
            onChange={handleInputChange}
            required
            placeholder="e.g., 7.5"
            disabled={loading}
          />
          <span className="input-hint">Hint: Most adults need 7-9 hours.</span>
        </div>
        {/* Exercise Frequency */}
        <div className="input-group">
          <label htmlFor="exerciseFreqInput">Exercise Frequency (days per week):</label>
          <input
            id="exerciseFreqInput"
            name="exerciseFreq"
            type="number"
            step="1"
            min="0"
            max="7"
            value={userInput.exerciseFreq}
            onChange={handleInputChange}
            required
            placeholder="e.g., 3"
            disabled={loading}
          />
          <span className="input-hint">Hint: Aim for 3-5 days of moderate activity.</span>
        </div>
         {/* Age */}
        <div className="input-group">
          <label htmlFor="ageInput">Age:</label>
          <input
            id="ageInput"
            name="age"
            type="number"
            min="1" 
            max="120" 
            value={userInput.age}
            onChange={handleInputChange}
            required
            placeholder="e.g., 35"
            disabled={loading}
          />
        </div>
        {/* Gender */}
         <div className="input-group">
          <label htmlFor="genderInput">Gender:</label>
          <select 
             id="genderInput"
             name="gender"
             value={userInput.gender}
             onChange={handleInputChange}
             required
             disabled={loading}
           >
            <option value="Male">Male</option>
            <option value="Female">Female</option>
           </select>
        </div>
         {/* Stress Level */}
        <div className="input-group">
          <label htmlFor="stressLevelInput">Average Stress Level (0-10):</label>
          <input
            id="stressLevelInput"
            name="stressLevel"
            type="range"
            min="0"
            max="10"
            step="1"
            value={userInput.stressLevel}
            onChange={handleInputChange}
            required
            disabled={loading}
          />
          <span>{userInput.stressLevel}</span>
          <span className="input-hint">Hint: 0 = No stress, 10 = Highest stress.</span>
        </div>
        
        <button type="submit" disabled={loading} className="submit-button primary-button">
          {loading ? 'Calculating...' : 'Get Prediction'}
        </button>
      </form>

      {loading && <p className="loading-message">Calculating prediction...</p>}
      {error && <p className="error-message">{error}</p>}

      {prediction && !loading && (
        <div className="prediction-result-display card-inset">
          <h3>Your Predicted Risk</h3>
           <div 
              className="risk-gauge" 
              style={{ 
                borderColor: getScoreColor(prediction.classification)
              }}
              title={`Risk Score: ${prediction.risk_score}`}
            >
                 <span className="risk-value" style={{ color: getScoreColor(prediction.classification) }}>
                    {prediction.risk_score}
                 </span>
            </div>
          <div className="result-details" style={{textAlign: 'center'}}>
             <p className="risk-classification">
                Classification: <span style={{fontWeight: 'bold', color: getScoreColor(prediction.classification)}}>{prediction.classification}</span>
             </p>
             <p className="risk-confidence">Model Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserInputSurvey; 