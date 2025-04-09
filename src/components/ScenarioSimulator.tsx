// react-app/src/components/ScenarioSimulator.tsx
import React, { useState, useEffect } from 'react';
import { fetchPrediction } from '../services/api';
import '../App.css'; // Import CSS

// Interface for the result data (should match backend)
interface SimulationResult {
  sleepHours: number;
  exerciseFreq: number;
  age: number; // Include age
  gender: string; // Include gender
  stressLevel: number; // Include stress
  risk_score: number;
  classification: string;
  confidence: number;
  timestamp?: string; // Optional timestamp
  iteratedVariable?: string; // Add a field to track which variable was iterated on (optional)
}

// Interface for initial user data passed from Dashboard
interface UserData {
  sleepHours: number | string;
  exerciseFreq: number | string;
  age: number | string;
  gender: string;
  stressLevel: number | string;
}

// Define props for ScenarioSimulator, including the callback and initial data
interface ScenarioSimulatorProps {
  onNewResult: (result: SimulationResult[]) => void;
  initialUserData: UserData | null; // Receive user's latest input
}

// UPDATED color mapping for 7 categories
const classificationColors: { [key: string]: string } = {
    "Extremely Low": '#198754',
    "Very Low": '#28a745',
    "Low": '#8fbc8f',
    "Moderate": '#FFC107',
    "High": '#fd7e14',
    "Very High": '#dc3545',
    "Extreme": '#8b0000',
    "Unknown": '#6c757d'
};

const ScenarioSimulator: React.FC<ScenarioSimulatorProps> = ({ onNewResult, initialUserData }) => {
  // State for Iterative Trend Simulation
  const [trendVariable, setTrendVariable] = useState<keyof UserData>('sleepHours'); // Variable to iterate
  const [trendStart, setTrendStart] = useState<number>(5);
  const [trendEnd, setTrendEnd] = useState<number>(9);
  const [trendStep, setTrendStep] = useState<number>(0.5);

  // State for Loading/Error
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // No more state needed for manual inputs or single latest result

  // Function for Iterative Trend Simulation
  const runIterativeSimulation = async (startVal = trendStart, endVal = trendEnd, stepVal = trendStep, variable = trendVariable) => {
      if (!initialUserData) {
          setError("Please submit the survey first to provide baseline data for simulation.");
          return;
      }
      
      setLoading(true);
      setError(null);
      
      const resultsBatch: SimulationResult[] = [];
      // Base parameters are taken directly from the user's latest input
      const baseParams = {
          sleepHours: Number(initialUserData.sleepHours),
          exerciseFreq: Number(initialUserData.exerciseFreq),
          age: Number(initialUserData.age),
          gender: initialUserData.gender,
          stressLevel: Number(initialUserData.stressLevel),
      };

      // Validate base params (ensure they are numbers after conversion)
      if (Object.values(baseParams).some(val => typeof val === 'number' && isNaN(val))) {
           setError("Invalid baseline data from survey input.");
           setLoading(false); return;
      }

      // Validate trend range
      if (isNaN(startVal) || isNaN(endVal) || isNaN(stepVal) || stepVal <= 0 || endVal < startVal) {
          setError("Invalid range/step for trend simulation.");
          setLoading(false); return;
      }

      try {
          for (let val = startVal; val <= endVal; val += stepVal) {
              const currentParams = { ...baseParams };
              let roundedVal = val; // Use original for sleep

              // Update the iterating variable and round if necessary
              if (variable === 'sleepHours') {
                   currentParams.sleepHours = val; 
              } else if (variable === 'exerciseFreq') {
                  roundedVal = Math.round(val);
                  currentParams.exerciseFreq = roundedVal;
              } else if (variable === 'stressLevel') {
                  roundedVal = Math.round(val);
                  currentParams.stressLevel = roundedVal;
              } else { 
                  throw new Error(`Invalid trend variable selected: ${variable}`);
              }
              
              // Validate the changing value within the loop bounds
              if (variable === 'sleepHours' && (val < 1 || val > 20)) continue; 
              if (variable === 'exerciseFreq' && (roundedVal < 0 || roundedVal > 7)) continue;
              if (variable === 'stressLevel' && (roundedVal < 0 || roundedVal > 10)) continue;

              // Fetch prediction using the specific parameters for this step
              const data = await fetchPrediction(currentParams);
              resultsBatch.push({
                  ...currentParams, // Store the exact params used for this iteration
                  risk_score: data.risk_score, 
                  classification: data.classification,
                  confidence: data.confidence,
                  timestamp: new Date().toISOString(),
                  iteratedVariable: variable // Mark which variable changed
              });
          }

          if (resultsBatch.length > 0) {
              onNewResult(resultsBatch); // Pass the whole batch up to Dashboard
              console.log(`Completed trend simulation for ${variable}, ${resultsBatch.length} steps.`);
          } else {
              setError("Trend simulation produced no results (check range/step or API limits).");
          }
      } catch (err) {
          console.error('Error running iterative simulation:', err);
          setError('Trend simulation failed. Check connection or backend.');
      }
      setLoading(false);
  };

  // Helper to get default step based on variable
  const getDefaultStep = (variable: keyof UserData): number => {
      switch(variable) {
          case 'sleepHours': return 0.5;
          case 'exerciseFreq': return 1;
          case 'stressLevel': return 1;
          default: return 1;
      }
  }

  // Update trend range/step when variable changes
  const handleTrendVariableChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      const newVar = e.target.value as keyof UserData;
      setTrendVariable(newVar);
      // Reset range/step to sensible defaults
      if (newVar === 'sleepHours') {
          setTrendStart(Math.max(4, (Number(initialUserData?.sleepHours) || 7) - 2)); 
          setTrendEnd(Math.min(12, (Number(initialUserData?.sleepHours) || 7) + 2)); 
          setTrendStep(0.5);
      } else if (newVar === 'exerciseFreq') {
          setTrendStart(0); setTrendEnd(7); setTrendStep(1);
      } else if (newVar === 'stressLevel') {
          setTrendStart(0); setTrendEnd(10); setTrendStep(1);
      }
  }
  
   // Preset Trend Simulation Handlers
   const runPresetTrend = (variable: keyof UserData, start: number, end: number, step: number) => {
       setTrendVariable(variable);
       setTrendStart(start);
       setTrendEnd(end);
       setTrendStep(step);
       // Run simulation immediately with these preset values
       runIterativeSimulation(start, end, step, variable);
   }

  return (
    <div className="scenario-simulator card-content">
       {/* Focus only on Iterative Trend Simulation */}
      <h2>Simulate Lifestyle Changes</h2>
        <p className="component-description">
            Select a factor (Sleep, Exercise, or Stress) and a range to simulate how changes 
            in that one area might impact your predicted risk score, based on your current survey data. 
            Focus on the <strong>direction and magnitude of change</strong> in the Trend Analysis chart, 
            as the absolute score may have limitations (see Model Info). 
            {!initialUserData && <strong style={{color: 'red'}}>(Requires survey submission first!)</strong>}
       </p>
       
       {/* Preset Trend Buttons */} 
       <div className="preset-buttons trend-presets">
           <button 
               onClick={() => runPresetTrend('sleepHours', Math.max(4, (Number(initialUserData?.sleepHours) || 7) - 2), Math.min(12, (Number(initialUserData?.sleepHours) || 7) + 2), 0.5)} 
               className="preset-button secondary-button" 
               disabled={!initialUserData || loading} 
               title="Simulate varying sleep around your current value"
            >
               Simulate Sleep Change
           </button>
           <button 
               onClick={() => runPresetTrend('exerciseFreq', 0, 7, 1)} 
               className="preset-button secondary-button" 
               disabled={!initialUserData || loading}
               title="Simulate varying exercise from 0 to 7 days/week"
            >
               Simulate Exercise Change
           </button>
            <button 
               onClick={() => runPresetTrend('stressLevel', 0, 10, 1)} 
               className="preset-button secondary-button" 
               disabled={!initialUserData || loading}
               title="Simulate varying stress from 0 to 10"
            >
               Simulate Stress Change
           </button>
       </div>
       
       <hr style={{margin: "20px 0"}} />
       
       <div className="simulator-controls trend-controls">
           <h4>Manual Trend Setup:</h4>
           {/* Select Variable */}
           <div className="input-group">
               <label htmlFor="trendVariable">Variable to Change:</label>
               <select id="trendVariable" value={trendVariable} onChange={handleTrendVariableChange} disabled={!initialUserData || loading}>
                   <option value="sleepHours">Sleep Hours</option>
                   <option value="exerciseFreq">Exercise Frequency</option>
                   <option value="stressLevel">Stress Level</option>
               </select>
           </div>
           {/* Range Inputs */}
           <div className="input-group range-inputs">
               <label htmlFor="trendStart">Start:</label>
               <input id="trendStart" type="number" value={trendStart} onChange={(e) => setTrendStart(Number(e.target.value))} step={getDefaultStep(trendVariable)} disabled={!initialUserData || loading} style={{width: '70px'}}/>
               <label htmlFor="trendEnd" style={{marginLeft: '10px'}}>End:</label>
               <input id="trendEnd" type="number" value={trendEnd} onChange={(e) => setTrendEnd(Number(e.target.value))} step={getDefaultStep(trendVariable)} disabled={!initialUserData || loading} style={{width: '70px'}}/>
               <label htmlFor="trendStep" style={{marginLeft: '10px'}}>Step:</label>
               <input id="trendStep" type="number" value={trendStep} onChange={(e) => setTrendStep(Number(e.target.value))} step="0.1" min="0.1" disabled={!initialUserData || loading} style={{width: '70px'}}/>
           </div>
           {/* Run Button */}
            <button onClick={() => runIterativeSimulation()} disabled={!initialUserData || loading} className="simulate-button primary-button">
                Run Manual Trend
            </button>
       </div>

      {/* Display Global loading or error state */}
      {loading && <p className="loading-message">Running simulation...</p>}
      {error && <p className="error-message">{error}</p>}
      
      {/* Removed the display for single simulation result */}

    </div>
  );
};

export default ScenarioSimulator;
