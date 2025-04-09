// react-app/src/components/TrendAnalysis.tsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Area, Legend } from 'recharts';

// Keep SimulationResult interface consistent
interface SimulationResult {
  sleepHours: number;
  exerciseFreq: number;
  risk_score: number;
  classification: string;
  confidence: number;
  timestamp?: string; 
}

// Define props for TrendAnalysis
interface TrendAnalysisProps {
  simulationHistory: SimulationResult[];
}

const TrendAnalysis: React.FC<TrendAnalysisProps> = ({ simulationHistory }) => {
  // Add a simple index or use timestamp if available for the x-axis
  const chartData = simulationHistory.map((result, index) => ({
    ...result,
    name: `Sim ${index + 1}` // Use simulation number as x-axis label
  }));

  return (
    <div className="trend-analysis-container">
      {/* Update title based on what's plotted */}
      <h2>Risk Score Trend (Last {chartData.length} Simulations)</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
            </linearGradient>
          </defs>
          {/* Update XAxis to use the generated name */}
          <XAxis dataKey="name" />
          {/* Update YAxis label and domain */}
          <YAxis 
            label={{ value: 'Risk Score', angle: -90, position: 'insideLeft' }} 
            domain={[0, 100]} // Risk score is 0-100
          />
          <Tooltip />
          <Legend />
          <CartesianGrid strokeDasharray="3 3" stroke="#eee"/>
          {/* Plot risk_score */}
          <Area type="monotone" dataKey="risk_score" name="Risk Score" stroke="#8884d8" fillOpacity={1} fill="url(#colorRisk)" />
          <Line type="monotone" dataKey="risk_score" name="Risk Score" stroke="#8884d8" strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 8 }} />
          {/* Optionally add another line for sleep hours or exercise freq for comparison */}
          {/* <Line type="monotone" dataKey="sleepHours" name="Sleep Hours" stroke="#82ca9d" /> */}
        </LineChart>
      </ResponsiveContainer>
      {simulationHistory.length === 0 && <p>Run simulations to see trends here.</p>}
    </div>
  );
};

export default TrendAnalysis;
