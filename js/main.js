async function getCropRecommendation(cropData) {

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(cropData)
        });

        if (!response.ok) {
            throw new Error("Failed to get crop recommendation");
        }

        const result = await response.json();

        console.log("Crop Recommendation:", result);

        return result;

    } catch (error) {
        console.error("Error:", error.message);
    }
}