// react-app/src/services/api.ts
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000'; // Default to backend server port

export const fetchPrediction = async (params: { sleepHours: number; exerciseFreq: number }) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      // Handle HTTP errors like 4xx, 5xx
      const errorData = await response.json().catch(() => ({})); // Attempt to get error details
      console.error('API Error:', response.status, errorData);
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Network or other error fetching prediction:', error);
    // Re-throw the error so UI components can handle it (e.g., show error message)
    throw error;
  }
};

// Add similar functions for other endpoints if needed, e.g., fetchSummary, fetchSimulation
export const fetchSummary = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/summary`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('API Error:', response.status, errorData);
            throw new Error(`API request failed with status ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching summary:', error);
        throw error;
    }
};

export const fetchCorrelation = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/correlation`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('API Error fetching correlation:', response.status, errorData);
            throw new Error(`API request failed with status ${response.status}`);
        }
        const data = await response.json();
        return data; // Expected format: { columns: string[], correlation: Record<string, any>[] }
    } catch (error) {
        console.error('Error fetching correlation data:', error);
        throw error;
    }
};
  