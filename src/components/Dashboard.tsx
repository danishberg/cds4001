// react-app/src/components/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import RiskScore from './RiskScore';
import TrendAnalysis from './TrendAnalysis';
import CorrelationAnalysis from './CorrelationAnalysis';
import ScenarioSimulator from './ScenarioSimulator';
import SummaryReport from './SummaryReport';
import UserInputSurvey from './UserInputSurvey';
import { fetchModelInfo } from '../services/api';
import '../App.css'; // Ensure App.css is imported for styles

// Keep SimulationResult interface consistent (should match ScenarioSimulator)
interface SimulationResult {
  sleepHours: number;
  exerciseFreq: number;
  age?: number;
  gender?: string;
  stressLevel?: number | string;
  risk_score: number;
  classification: string;
  confidence: number;
  timestamp?: string;
  iteratedVariable?: string; // Include the optional iterated variable field
}

// Interfaces (keep SimulationResult and define others if needed)
interface UserData {
  sleepHours: number | string;
  exerciseFreq: number | string;
  age: number | string;
  gender: string;
  stressLevel: number | string;
}

interface PredictionResult {
  risk_score: number;
  classification: string;
  confidence: number;
}

// Interface for model info from API
interface ModelInfo {
    accuracy: number;
    features_used: string[];
    target: string;
    model_type: string;
    error?: string;
}

// Helper function to format feature names (moved here from planned component)
const formatFeatureName = (name: string): string => {
    return name
        .replace('sleep_duration', 'Sleep Duration')
        .replace('exercise_freq', 'Exercise Frequency')
        .replace('Gender_encoded', 'Gender')
        .replace(/([A-Z])/g, ' $1') // Add space before caps
        .replace(/^./, str => str.toUpperCase()); // Capitalize first letter
};

const Dashboard: React.FC = () => {
  // State to hold the simulation results history, managed by Dashboard
  const [simulationHistory, setSimulationHistory] = useState<SimulationResult[]>([]);
  // State for the last user survey input and its prediction
  const [lastUserInput, setLastUserInput] = useState<UserData | null>(null);
  const [lastPrediction, setLastPrediction] = useState<PredictionResult | null>(null);
  // State for model info
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelInfoError, setModelInfoError] = useState<string | null>(null);

  // Fetch Model Info on mount
  useEffect(() => {
    const getModelInfo = async () => {
        const info = await fetchModelInfo();
        if (info.error) {
            setModelInfoError(info.error);
            console.error("Error fetching model info:", info.error);
        } else {
            setModelInfo(info);
        }
    };
    getModelInfo();
  }, []); // Empty dependency array means run once on mount

  // SIMPLIFIED Callback - Now only expects arrays from ScenarioSimulator
  const handleNewSimulationResult = (resultsBatch: SimulationResult[]) => {
    // Directly replace the history with the new batch from trend simulation
    setSimulationHistory(resultsBatch);
    console.log("Dashboard: Replaced history with trend simulation batch:", resultsBatch);
    // Removed the logic for handling single results
  };

  // Callback for UserInputSurvey
  const handleNewUserInputPrediction = (input: UserData, result: PredictionResult) => {
    setLastUserInput(input);
    setLastPrediction(result);
    console.log("User Input Prediction Received:", input, result);
  };

  return (
    <div className="dashboard-container">
      <h1 className="dashboard-title">Sleep & Lifestyle Health Dashboard</h1>

      <div className="dashboard-grid">
        <div className="dashboard-card large-card user-input-card">
          <UserInputSurvey onPredictionComplete={handleNewUserInputPrediction} />
        </div>
        <div className="dashboard-card risk-score-card">
          <RiskScore prediction={lastPrediction} />
        </div>
        <div className="dashboard-card summary-report-card">
          <SummaryReport 
              userInput={lastUserInput} 
              prediction={lastPrediction} 
              modelInfo={modelInfo}
          />
        </div>
        <div className="dashboard-card model-info-card">
            <div className="model-info-container card-content">
                 {modelInfoError ? (
                    <>
                        <h2>About the Prediction Model</h2>
                        <p className="error-message">Could not load model details: {modelInfoError}</p>
                    </>
                 ) : !modelInfo ? (
                    <>
                        <h2>About the Prediction Model</h2>
                        <p className="loading-message">Loading model details...</p>
                    </>
                 ) : (
                    <>
                        <h2>About the Prediction Model</h2>
                        <div className="model-details">
                            <p><strong>Type:</strong> {modelInfo.model_type}</p>
                            <p><strong>Accuracy:</strong> {(modelInfo.accuracy * 100).toFixed(1)}%</p>
                            <p><small>(Accuracy reflects how often the model's risk classification matched the label in the test dataset during training.)</small></p>
                            <p><strong>Features Used:</strong> {modelInfo.features_used.map(formatFeatureName).join(', ')}</p>
                        </div>
                        <div className="model-limitations">
                            <h4>Important Limitations:</h4>
                            <ul>
                                <li>This is a simplified statistical model, not a comprehensive health assessment.</li>
                                <li>It predicts risk based *only* on the factors listed above. Many other factors influence health.</li>
                                <li><strong>This is NOT a medical diagnosis.</strong> Always consult a qualified healthcare professional for medical advice.</li>
                                <li>The model identifies correlations in data, which does not necessarily imply causation.</li>
                            </ul>
                        </div>
                    </>
                 )}
            </div>
        </div>
        <div className="dashboard-card trend-analysis-card">
          <TrendAnalysis 
              simulationHistory={simulationHistory} 
              baselinePrediction={lastPrediction}
           />
        </div>
        <div className="dashboard-card large-card simulator-card">
          <ScenarioSimulator 
              onNewResult={handleNewSimulationResult} // Handler now expects SimulationResult[]
              initialUserData={lastUserInput} 
          />
        </div>
        <div className="dashboard-card correlation-card">
          <CorrelationAnalysis />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
