// react-app/src/services/api.ts
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000'; // Default to backend server port

// Define the full input structure
interface PredictionParams {
  sleepHours: number;
  exerciseFreq: number;
  age?: number;       // Optional for initial prediction
  gender?: string;    // Optional for initial prediction
  stressLevel?: number | string; // Optional for initial prediction
}

// Define the expected structure for model info
interface ModelInfo {
    accuracy: number;
    features_used: string[];
    target: string;
    model_type: string;
    error?: string; // Include potential error message
}

export const fetchPrediction = async (params: PredictionParams) => {
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

// NEW function to fetch model info
export const fetchModelInfo = async (): Promise<ModelInfo> => {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: "Unknown error fetching model info" }));
            console.error('API Error fetching model info:', response.status, errorData);
            throw new Error(errorData.error || `API request failed with status ${response.status}`);
        }
        const data: ModelInfo = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching model info:', error);
        // Return an object indicating error for the UI to handle
        return { accuracy: 0, features_used: [], target: '', model_type: '', error: error instanceof Error ? error.message : String(error) };
    }
};
  