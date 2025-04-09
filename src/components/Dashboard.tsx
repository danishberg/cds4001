// react-app/src/components/Dashboard.tsx
import React from 'react';
import RiskScore from './RiskScore';
import TrendAnalysis from './TrendAnalysis';
import CorrelationAnalysis from './CorrelationAnalysis';
import ScenarioSimulator from './ScenarioSimulator';
import SummaryReport from './SummaryReport';

const Dashboard: React.FC = () => {
  return (
    <div style={{ padding: '20px' }}>
      <h1>Health Outcome Prediction Dashboard</h1>
      <RiskScore />
      <TrendAnalysis />
      <CorrelationAnalysis />
      <ScenarioSimulator />
      <SummaryReport />
    </div>
  );
};

export default Dashboard;
