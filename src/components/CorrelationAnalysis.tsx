// react-app/src/components/CorrelationAnalysis.tsx
import React, { useState, useEffect } from 'react';
import { fetchCorrelation } from '../services/api';
import '../App.css';

interface CorrelationResponse {
  columns: string[];
  correlation: Record<string, any>[];
}

const CorrelationAnalysis: React.FC = () => {
  const [data, setData] = useState<CorrelationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCorrelation()
      .then(responseData => {
        setData(responseData);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching correlation:", err);
        setError("Failed to load correlation data.");
        setLoading(false);
      });
  }, []);

  const getCorrelationColor = (value: number): string => {
    if (isNaN(value)) return '#ccc'; // Grey for NaN
    // Simple scale: Red for negative, Blue for positive
    const intensity = Math.min(Math.abs(value) * 200, 150); // Cap intensity
    if (value > 0) {
      return `rgba(0, 0, 255, ${Math.abs(value).toFixed(2)})`; // Blue intensity
    } else if (value < 0) {
      return `rgba(255, 0, 0, ${Math.abs(value).toFixed(2)})`; // Red intensity
    } else {
      return '#fff'; // White for zero
    }
  };

  return (
    <div className="correlation-analysis-container">
      <h2>Correlation Analysis (Numeric Features)</h2>
      {loading && <p>Loading correlation matrix...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {data && (
        <div className="correlation-table-container">
          <table className="correlation-table">
            <thead>
              <tr>
                <th>Variable</th>
                {data.columns.map(col => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.correlation.map((row, rowIndex) => (
                <tr key={row.index || rowIndex}> {/* Use index from data if available */}
                  <td>{row.index}</td>
                  {data.columns.map(col => {
                    const value = row[col];
                    return (
                      <td 
                        key={col}
                        style={{ 
                          backgroundColor: getCorrelationColor(value),
                          color: Math.abs(value) > 0.5 ? 'white' : 'black' // Adjust text color for readability
                        }}
                        title={`Correlation(${row.index}, ${col}): ${typeof value === 'number' ? value.toFixed(3) : 'N/A'}`}
                      >
                        {typeof value === 'number' ? value.toFixed(2) : '-'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default CorrelationAnalysis;
