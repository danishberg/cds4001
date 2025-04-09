// react-app/src/components/TrendAnalysis.tsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Area, Legend, TooltipProps, ReferenceDot } from 'recharts';
import { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent';

// Interfaces
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
  iteratedVariable?: string; 
}
// Define PredictionResult interface locally
interface PredictionResult { 
  risk_score: number;
  classification: string;
  confidence: number;
}

// Update props to include baselinePrediction
interface TrendAnalysisProps {
  simulationHistory: SimulationResult[];
  baselinePrediction: PredictionResult | null; // User's prediction before simulation
}

// Custom Tooltip (adjust label slightly)
const CustomTooltip: React.FC<TooltipProps<ValueType, NameType>> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as SimulationResult;
    // Use the iterated variable name in the label if available
    const labelPrefix = data.iteratedVariable ? 
        data.iteratedVariable.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()) + ": " 
        : "Index: ";
    return (
      <div className="custom-tooltip card-inset" style={{ backgroundColor: 'rgba(255, 255, 255, 0.9)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}>
        <p className="label">{`${labelPrefix}${label}`}</p> 
        <p className="intro" style={{color: payload[0].color }}>{`Risk Score: ${data.risk_score}`}</p>
        <p style={{fontSize: '0.8em', color: '#666'}}>{`Classification: ${data.classification}`}</p>
      </div>
    );
  }
  return null;
};

const TrendAnalysis: React.FC<TrendAnalysisProps> = ({ simulationHistory, baselinePrediction }) => {
  const iteratedVar = simulationHistory.length > 0 ? simulationHistory[0].iteratedVariable : null;
  const titleVariable = iteratedVar ? 
      iteratedVar.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()) 
      : "Simulation History";
  const xAxisLabel = iteratedVar ? titleVariable : "Simulation Index";
  const xAxisDataKey = iteratedVar || "name"; 

  // Map data for the chart
  const chartData = simulationHistory.map((result, index) => ({
    ...result,
    name: `Sim ${index + 1}`, 
    // Use the actual iterated value or index as the key for the x-axis
    [xAxisDataKey]: iteratedVar ? result[iteratedVar as keyof SimulationResult] : index + 1
  }));

  // Determine title
  let chartTitle = `Risk Score Trend (${chartData.length} Points)`;
  if (iteratedVar) {
      chartTitle = `Simulated Risk Score vs. ${titleVariable}`;
  }

  // --- Reference Dot Logic (Revised) ---
  let baselineRefDotProps: { x: number | string, y: number } | null = null;
  if (iteratedVar && baselinePrediction && chartData.length > 0) {
      const firstPoint = chartData[0];
      // Get the potential x value
      const xValue = firstPoint[xAxisDataKey as keyof typeof firstPoint];
      
      // Ensure xValue is defined and is a string or number before creating props
      if (xValue !== undefined && (typeof xValue === 'string' || typeof xValue === 'number')) {
          baselineRefDotProps = {
              x: xValue, 
              y: baselinePrediction.risk_score
          };
      } 
      // If xValue is undefined, baselineRefDotProps remains null, and the dot won't render
  }

  return (
    <div className="trend-analysis-container card-content">
      <h2>{chartTitle}</h2>
      {simulationHistory.length === 0 ? (
          <p className="report-placeholder">Run a simulation to see trends here.</p>
      ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 10, bottom: 25 }}> {/* Increased bottom margin slightly */}
              <defs>
                <linearGradient id="colorRiskArea" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8884d8" stopOpacity={0.7}/>
                  <stop offset="95%" stopColor="#8884d8" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <XAxis 
                dataKey={xAxisDataKey} 
                type="number" // Keep assuming numeric iterated variables for axis type
                domain={['dataMin', 'dataMax']} 
                label={iteratedVar ? { value: xAxisLabel, position: 'insideBottom', dy: 15 } : undefined} 
                tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(1) : tick} 
                allowDuplicatedCategory={false} // Prevent overlapping labels if steps are small
              />
              <YAxis 
                label={{ value: 'Risk Score', angle: -90, position: 'insideLeft' }} 
                domain={[0, 100]} 
                allowDataOverflow={true}
              />
              <Tooltip content={<CustomTooltip />} /> 
              <Legend wrapperStyle={{paddingTop: '15px'}} /> {/* Add padding above legend */}
              <CartesianGrid strokeDasharray="3 3" stroke="#eee"/>
              <Area 
                  type="monotone" 
                  dataKey="risk_score" 
                  name="Risk Score" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorRiskArea)" 
              />
              <Line 
                  type="monotone" 
                  dataKey="risk_score" 
                  name="Risk Score" 
                  stroke="#8884d8" 
                  strokeWidth={1} 
                  dot={{ r: 4 }} 
                  activeDot={{ r: 6 }} 
              />

              {/* Add ReferenceDot using simplified props */}
              {baselineRefDotProps && (
                    <ReferenceDot 
                        x={baselineRefDotProps.x} 
                        y={baselineRefDotProps.y} 
                        r={6} 
                        fill="#FF4500" 
                        stroke="white" 
                        strokeWidth={2}
                        label={{ 
                            value: "Your Baseline", 
                            position: "top", 
                            fill: "#FF4500", 
                            fontSize: 12 
                        }}
                        isFront={true}
                    />
              )}
            </LineChart>
          </ResponsiveContainer>
      )}
    </div>
  );
};

export default TrendAnalysis;
