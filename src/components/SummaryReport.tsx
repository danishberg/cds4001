// react-app/src/components/SummaryReport.tsx
import React, { useState, useEffect } from 'react';
import { fetchSummary } from '../services/api'; // Import the fetchSummary function
import '../App.css';

interface SummaryData {
  sleep_duration?: { mean: number; median: number; min: number; max: number };
  exercise_freq?: { mean: number; median: number; min: number; max: number };
  risk_distribution?: { [key: string]: number };
}

const SummaryReport: React.FC = () => {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSummary()
      .then(data => {
        setSummary(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching summary:", err);
        setError("Failed to load summary data.");
        setLoading(false);
      });
  }, []);

  return (
    <div className="summary-report-container">
      <h2>Dataset Summary Report</h2>
      {loading && <p>Loading summary...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {summary && (
        <div className="summary-details">
          {summary.sleep_duration && (
            <div className="summary-section">
              <h3>Sleep Duration (hours)</h3>
              <ul>
                <li>Mean: {summary.sleep_duration.mean}</li>
                <li>Median: {summary.sleep_duration.median}</li>
                <li>Min: {summary.sleep_duration.min}</li>
                <li>Max: {summary.sleep_duration.max}</li>
              </ul>
            </div>
          )}
          {summary.exercise_freq && (
            <div className="summary-section">
              <h3>Exercise Frequency (days/week)</h3>
               <ul>
                <li>Mean: {summary.exercise_freq.mean}</li>
                <li>Median: {summary.exercise_freq.median}</li>
                <li>Min: {summary.exercise_freq.min}</li>
                <li>Max: {summary.exercise_freq.max}</li>
              </ul>
            </div>
          )}
          {summary.risk_distribution && (
            <div className="summary-section">
              <h3>Risk Distribution</h3>
               <ul>
                {Object.entries(summary.risk_distribution).map(([key, value]) => (
                  <li key={key}>{key}: {value} users</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SummaryReport;
