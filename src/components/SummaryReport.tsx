// react-app/src/components/SummaryReport.tsx
import React from 'react';
import '../App.css';

// Interfaces for props (should match Dashboard)
interface UserData { 
    sleepHours: number | string;
    exerciseFreq: number | string;
    age: number | string;
    gender: string;
    stressLevel: number | string;
}
interface PredictionResult { 
  risk_score: number;
  classification: string;
  confidence: number;
}
// Add ModelInfo interface (or import if defined centrally)
interface ModelInfo {
    accuracy: number;
    features_used: string[];
    target: string;
    model_type: string;
    error?: string;
}

interface SummaryReportProps {
    userInput: UserData | null;
    prediction: PredictionResult | null;
    modelInfo: ModelInfo | null; // Accept modelInfo
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

// Enhanced helper function for report generation
const generateHealthInsights = (input: UserData | null, pred: PredictionResult | null, modelInfo: ModelInfo | null): string[] => {
    const insights: string[] = [];
    if (!input || !pred) return ["Submit your data in the survey to generate a personalized report."];

    // Convert to numbers for easier comparison
    const sleep = Number(input.sleepHours);
    const exercise = Number(input.exerciseFreq);
    const stress = Number(input.stressLevel);
    const age = Number(input.age);

    // ... input conversions ...
    const featuresUsedString = modelInfo?.features_used ? modelInfo.features_used.join(', ').replace('Gender_encoded', 'Gender').replace('sleep_duration', 'Sleep Duration').replace('exercise_freq', 'Exercise Frequency').trim() : 'the model inputs';

    // --- UPDATED General Comment based on Classification (7 categories) ---
    switch (pred.classification) {
        case 'Extremely Low':
            insights.push(`✅✅ Overall: Your predicted health risk is Extremely Low. Based on the model and your inputs (${featuresUsedString}), your current lifestyle appears exceptionally supportive of good health.`);
            break;
        case 'Very Low':
            insights.push(`✅ Overall: Your predicted health risk is Very Low. This suggests your current lifestyle habits related to ${featuresUsedString} are highly supportive of good health.`);
            break;
        case 'Low':
            insights.push(`✅ Overall: Your predicted health risk is Low. This suggests your habits related to ${featuresUsedString} are generally supportive of good health. Keep it up!`);
            break;
        case 'Moderate':
            insights.push(`⚠️ Overall: Your predicted health risk is Moderate. The model suggests potential opportunities for improvement in areas like sleep, exercise, or stress management.`);
            break;
        case 'High':
            insights.push(`❗ Overall: Your predicted health risk is High. The model indicates that factors like sleep, exercise, and stress might be contributing to elevated risk. Consider focusing on improvements.`);
            break;
        case 'Very High':
            insights.push(`❗❗ Overall: Your predicted health risk is Very High. The model indicates that current factors (especially sleep, exercise, stress) pose a significant risk. Lifestyle changes are strongly recommended.`);
            break;
        case 'Extreme':
             insights.push(`🚨 Overall: Your predicted health risk is Extreme. Based on the model, current factors indicate a critical risk level. Consulting a healthcare professional and making significant lifestyle changes is urgently advised.`);
             break;
        default:
            insights.push("Overall risk assessment based on current inputs.");
    }

    // --- Specific Insights based on Input Values & Combinations ---
    
    // Sleep Insights
    if (sleep < 6) {
        insights.push("🛌 Sleep: Getting less than 6 hours of sleep regularly is often associated with increased health risks. Aiming for 7-9 hours is generally recommended.");
    } else if (sleep > 9) {
        // insights.push("Sleep: While sufficient sleep is good, consistently sleeping much longer than 9 hours might warrant discussion with a doctor, though it's less commonly linked to risk in simple models.");
    } else {
        insights.push("🛌 Sleep: Your sleep duration (7-9 hours) appears to be within the generally recommended range.");
    }

    // Exercise Insights
    if (exercise < 3) {
        insights.push("🏃 Exercise: Exercising fewer than 3 days a week might be an area for improvement. Increasing physical activity, even moderately, can significantly benefit health.");
    } else if (exercise >= 5) {
        insights.push("🏃 Exercise: Exercising 5 or more days a week is excellent! Keep up the great work maintaining a high activity level.");
    } else {
         insights.push("🏃 Exercise: Your reported exercise frequency (3-4 days/week) is beneficial.");
    }

    // Stress Insights
    if (stress >= 8) {
         insights.push("🧘 Stress: High stress levels (8+) can significantly impact sleep quality and overall health. Exploring stress management techniques (like mindfulness, hobbies, or seeking support) could be beneficial.");
    } else if (stress <= 3) {
         insights.push("🧘 Stress: Low reported stress levels (0-3) are generally positive for health. Continue managing stress effectively.");
    } else {
         insights.push("🧘 Stress: Your reported stress level seems moderate (4-7).");
    }
    
    // --- Combination Insights (Examples) ---
    if (sleep < 6 && stress >= 8) {
        insights.push("💡 Combination: The combination of low sleep (< 6 hours) and high stress (8+) can be particularly challenging for health. Improving sleep might help manage stress, and vice-versa.");
    }
     if (exercise < 3 && stress >= 7) {
        insights.push("💡 Combination: Low exercise frequency (< 3 days) combined with high stress (7+) might increase risk. Physical activity can be an effective stress reliever.");
    }

    // --- Age/Gender Comments (Informational) ---
    // insights.push(`ℹ️ Demographics: Factors like age (${input.age}) and gender (${input.gender}) are included in the model as they can influence health outcomes, although lifestyle factors are often more modifiable.`);
    
    // --- Dynamic Disclaimer ---
    const modelFeats = modelInfo?.features_used;
    const featuresString = modelFeats 
        ? modelFeats
            .map(name => name
                .replace('sleep_duration', 'Sleep Duration')
                .replace('exercise_freq', 'Exercise Frequency')
                .replace('Gender_encoded', 'Gender')
                .replace('Stress Level', 'Stress') // Shorten if needed
                .replace(/([A-Z])/g, ' $1')
                .replace(/^./, str => str.toUpperCase())
                .trim()
            )
            .join(', ') 
        : "the available inputs";
        
    const accuracyString = modelInfo ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : "N/A";
        
    insights.push(`
        <hr style="border-top: 1px dashed #ccc; margin: 10px 0;" />
        <strong>Disclaimer:</strong> This tool provides a simplified risk prediction based on a statistical model using factors like ${featuresString}. 
        Overall model accuracy on test data is estimated at <strong>${accuracyString}</strong>. 
        Individual prediction confidence was ${(pred.confidence * 100).toFixed(1)}%. 
        This is <strong>not</strong> a medical diagnosis. Correlation does not equal causation. 
        Always consult a qualified healthcare professional for personalized medical advice and diagnosis.
    `);

    return insights;
}

const SummaryReport: React.FC<SummaryReportProps> = ({ userInput, prediction, modelInfo }) => {

  // Pass modelInfo directly to the helper function
  const insights = generateHealthInsights(userInput, prediction, modelInfo);
  const classificationColor = prediction ? (classificationColors[prediction.classification] || classificationColors.Unknown) : classificationColors.Unknown;

  return (
    <div className="summary-report-container card-content">
      <h2>Your Personal Health Summary</h2>
      
      {!userInput || !prediction ? (
         <p className="report-placeholder">{insights[0]}</p>
      ) : (
        <div className="report-content">
          <div className="report-header" style={{ borderLeftColor: classificationColor }}>
             <h4>Overall Risk Assessment</h4>
             {/* Updated to include all inputs */}
             <p>Based on your input (Sleep: {userInput.sleepHours} hrs, Exercise: {userInput.exerciseFreq} days/wk, Age: {userInput.age}, Gender: {userInput.gender}, Stress: {userInput.stressLevel}), your predicted risk category is: 
                <strong style={{ color: classificationColor }}> {prediction.classification}</strong>.
             </p>
             <p>(Model Confidence: {(prediction.confidence * 100).toFixed(1)}%)</p>
          </div>

          <div className="report-insights">
            <h4>Key Insights & Recommendations:</h4>
            <ul>
              {insights.map((insight, index) => (
                  // Use dangerouslySetInnerHTML for the disclaimer containing HTML
                  insight.includes("<hr") ? 
                  <li key={index} dangerouslySetInnerHTML={{ __html: insight }}></li> :
                  <li key={index}>{insight}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default SummaryReport;
