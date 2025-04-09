// react-app/src/components/Dashboard.tsx
import React, { useState } from 'react';
import RiskScore from './RiskScore';
import TrendAnalysis from './TrendAnalysis';
import CorrelationAnalysis from './CorrelationAnalysis';
import ScenarioSimulator from './ScenarioSimulator';
import SummaryReport from './SummaryReport';
import UserInputSurvey from './UserInputSurvey';
import '../App.css'; // Ensure App.css is imported for styles

// Define the structure for simulation results, consistent with ScenarioSimulator
interface SimulationResult {
  sleepHours: number;
  exerciseFreq: number;
  risk_score: number;
  classification: string;
  confidence: number;
  timestamp?: string; // Add timestamp if used in simulator
}

const Dashboard: React.FC = () => {
  // State to hold the simulation results history, managed by Dashboard
  const [simulationHistory, setSimulationHistory] = useState<SimulationResult[]>([]);

  // Callback function for ScenarioSimulator to update the history
  const handleNewSimulationResult = (result: SimulationResult) => {
    setSimulationHistory(prev => [...prev, result].slice(-10)); // Keep last 10 results
  };

  // Optional: Callback for the user input prediction (can be used later)
  // const handleNewUserInputPrediction = (result: any) => {
  //   console.log("User Input Prediction:", result);
  // };

  return (
    <div className="dashboard-container">
      <h1 className="dashboard-title">Health Outcome Prediction Dashboard</h1>
      <div className="dashboard-grid">
        <div className="dashboard-card large-card">
          <UserInputSurvey /* onNewPrediction={handleNewUserInputPrediction} */ />
        </div>
        <div className="dashboard-card">
          <RiskScore />
        </div>
        <div className="dashboard-card">
          <TrendAnalysis simulationHistory={simulationHistory} />
        </div>
        <div className="dashboard-card large-card">
          <ScenarioSimulator onNewResult={handleNewSimulationResult} />
        </div>
        <div className="dashboard-card">
          <CorrelationAnalysis />
        </div>
        <div className="dashboard-card">
          <SummaryReport />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
