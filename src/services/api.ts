// react-app/src/services/api.ts
export const fetchPrediction = async (params: { sleepHours: number; exerciseFreq: number }) => {
    const response = await fetch('http://localhost:3000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    const data = await response.json();
    return data;
  };
  