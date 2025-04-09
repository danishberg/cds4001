// react-app/src/components/ScenarioSimulator.tsx
import React, { useState } from 'react';
import { fetchPrediction } from '../services/api';

const ScenarioSimulator: React.FC = () => {
  const [sleepHours, setSleepHours] = useState(7);
  const [exerciseFreq, setExerciseFreq] = useState(3);
  const [result, setResult] = useState<any>(null);

  const runSimulation = () => {
    fetchPrediction({ sleepHours, exerciseFreq }).then(data => {
      setResult(data);
    });
  };

  return (
    <div>
      <h2>Scenario Simulator</h2>
      <div>
        <label>
          Sleep Hours:
          <input type="number" value={sleepHours} onChange={(e) => setSleepHours(Number(e.target.value))} />
        </label>
      </div>
      <div>
        <label>
          Exercise Frequency:
          <input type="number" value={exerciseFreq} onChange={(e) => setExerciseFreq(Number(e.target.value))} />
        </label>
      </div>
      <button onClick={runSimulation}>Run Simulation</button>
      {result && (
        <div>
          <p>Risk Score: {result.risk_score}</p>
          <p>Classification: {result.classification}</p>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
};

export default ScenarioSimulator;
