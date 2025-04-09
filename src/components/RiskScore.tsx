// react-app/src/components/RiskScore.tsx
import React from 'react'; // Removed useState, useEffect
// Removed fetchPrediction import
import '../App.css'; // Import CSS for styling

// Interface for the prediction data passed as a prop
interface PredictionResult { 
  risk_score: number;
  classification: string;
  confidence: number;
}

// Define props for RiskScore
interface RiskScoreProps {
    prediction: PredictionResult | null; // Accept prediction or null
}

// UPDATED color mapping for 7 categories
const classificationColors: { [key: string]: string } = {
    "Extremely Low": '#198754',// Bootstrap Success Darker
    "Very Low": '#28a745',   // Bootstrap Success
    "Low": '#8fbc8f',      // DarkSeaGreen 
    "Moderate": '#FFC107', // Amber
    "High": '#fd7e14',      // Bootstrap Orange
    "Very High": '#dc3545',  // Bootstrap Danger
    "Extreme": '#8b0000',    // DarkRed
    "Unknown": '#6c757d'    // Grey
};

const RiskScore: React.FC<RiskScoreProps> = ({ prediction }) => {
  // No internal state or useEffect needed anymore

  // Use the shared color map based on classification
  const getScoreColor = (classification: string | undefined) => {
    if (!classification) return classificationColors.Unknown;
    return classificationColors[classification] || classificationColors.Unknown;
  };

  // Determine color based on the passed prediction
  const scoreColor = getScoreColor(prediction?.classification);

  return (
    <div className="risk-score-container card-content"> {/* Added card-content */} 
      <h2>Your Current Risk Score</h2>
      {prediction ? (
        <div className="risk-details">
          <div 
            className="risk-gauge" 
            style={{ borderColor: scoreColor }}
            title={`Risk Score: ${prediction.risk_score} | Classification: ${prediction.classification}`}
          >
            <span className="risk-value" style={{ color: scoreColor }}>
              {prediction.risk_score}
            </span>
          </div>
          <p className="risk-classification">
            Classification: <span style={{ fontWeight: 'bold', color: scoreColor }}>{prediction.classification}</span>
          </p>
          <p className="risk-confidence">Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
        </div>
      ) : (
        <p className="report-placeholder">Submit the survey to see your risk score here.</p> // Show placeholder if no prediction
      )}
    </div>
  );
};

export default RiskScore;
