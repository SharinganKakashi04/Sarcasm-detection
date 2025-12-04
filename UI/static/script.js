document.addEventListener('DOMContentLoaded', () => {
    const analyzeButton = document.querySelector('.analyze-button');
    const textArea = document.querySelector('.text-input');
    const outputArea = document.getElementById('output-area');

    analyzeButton.addEventListener('click', async () => {
        const text = textArea.value.trim();
        if (text === "") {
            outputArea.innerHTML = "<p style='color:red;'>Please enter some text.</p>";
            return;
        }

        outputArea.innerHTML = "<p>Analyzing...</p>";

        try {
            // 1. Send the text to the Flask /predict API endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const result = await response.json();

            // 2. Format the output based on the result
            const confidenceSarcastic = result.confidence.sarcastic * 100;
            const confidenceNonSarcastic = result.confidence['non-sarcastic'] * 100;
            
            const htmlOutput = `
                <div class="result-box">
                    <p class="result-text">Text: ${result.text}</p>
                    <hr>
                    <p class="result-prediction">Prediction: ${result.prediction}</p>
                    <p class="result-confidence">Confidence Scores:</p>
                    <ul>
                        <li>Non-sarcastic: ${confidenceNonSarcastic.toFixed(2)}%</li>
                        <li>Sarcastic: ${confidenceSarcastic.toFixed(2)}%</li>
                    </ul>

                    <div class="feedback-buttons">
                        <p>Was this prediction correct?</p>
                        <button class="feedback-btn" data-correct-label="correct">✅ Correct</button>
                        <button class="feedback-btn" data-correct-label="wrong">❌ Wrong</button>
                    </div>
                </div>
            `;
            
            // 4. Inject the final HTML into the empty space
            outputArea.innerHTML = htmlOutput;

            // TODO: Add event listeners for the new feedback buttons here
            // This is the next phase (connecting the feedback)

        } catch (error) {
            console.error('Prediction error:', error);
            outputArea.innerHTML = "<p style='color:red;'>Error connecting to model server.</p>";
        }
    });
});