// react-app/src/components/TrendAnalysis.tsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const sampleData = [
  { day: 'Mon', sleep: 7 },
  { day: 'Tue', sleep: 6 },
  { day: 'Wed', sleep: 8 },
  { day: 'Thu', sleep: 5 },
  { day: 'Fri', sleep: 7 },
  { day: 'Sat', sleep: 9 },
  { day: 'Sun', sleep: 8 },
];

const TrendAnalysis: React.FC = () => {
  return (
    <div>
      <h2>Trend Analysis</h2>
      <LineChart width={600} height={300} data={sampleData}>
        <XAxis dataKey="day" />
        <YAxis label={{ value: 'Sleep Hours', angle: -90, position: 'insideLeft' }}/>
        <Tooltip />
        <CartesianGrid strokeDasharray="3 3" />
        <Line type="monotone" dataKey="sleep" stroke="#8884d8" />
      </LineChart>
    </div>
  );
};

export default TrendAnalysis;
