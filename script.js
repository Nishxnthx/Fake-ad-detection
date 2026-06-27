async function predictAd() {
    const adText = document.getElementById("adText").value.trim();
    const loadingText = document.getElementById("loadingText");
    const summaryCard = document.getElementById("summaryCard");
    const dashboardSection = document.getElementById("dashboardSection");

    if (adText === "") {
        alert("Please enter advertisement text.");
        return;
    }

    loadingText.innerText = "Running complete ML analysis...";
    summaryCard.classList.add("hidden");
    dashboardSection.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                ad_text: adText
            })
        });

        const data = await response.json();

        if (!response.ok) {
            alert(data.error || "Something went wrong.");
            loadingText.innerText = "";
            return;
        }

        // Final summary section
        document.getElementById("finalPrediction").innerText = data.final_prediction;
        document.getElementById("finalConfidence").innerText = data.confidence + "%";
        document.getElementById("finalRiskLevel").innerText = data.risk_level;
        document.getElementById("finalAction").innerText = data.user_action;

        // Neural Network section
        document.getElementById("nnPrediction").innerText = data.neural_analysis.prediction;
        document.getElementById("nnConfidence").innerText = data.neural_analysis.confidence + "%";
        document.getElementById("nnExplanation").innerText = data.neural_analysis.explanation;

        // Regression section
        document.getElementById("riskScore").innerText = data.regression_analysis.risk_score + "%";
        document.getElementById("riskLevel").innerText = data.regression_analysis.risk_level;
        document.getElementById("riskExplanation").innerText = data.regression_analysis.explanation;

        const riskBar = document.getElementById("riskBar");
        riskBar.style.width = data.regression_analysis.risk_score + "%";

        // Clustering section
        document.getElementById("clusterNumber").innerText = data.clustering_analysis.cluster_number;
        document.getElementById("clusterName").innerText = data.clustering_analysis.cluster_name;
        document.getElementById("clusterExplanation").innerText = data.clustering_analysis.explanation;

        // Recommendation section
        document.getElementById("userAction").innerText = data.recommendation_analysis.user_action;
        document.getElementById("recommendationText").innerText = data.recommendation_analysis.recommendation;
        document.getElementById("recommendationExplanation").innerText = data.recommendation_analysis.explanation;

        // Apply color classes
        applyPredictionColor("finalPrediction", data.final_prediction);
        applyPredictionColor("nnPrediction", data.neural_analysis.prediction);

        applyRiskColor("finalRiskLevel", data.risk_level);
        applyRiskColor("riskLevel", data.regression_analysis.risk_level);

        summaryCard.classList.remove("hidden");
        dashboardSection.classList.remove("hidden");
        loadingText.innerText = "";

    } catch (error) {
        loadingText.innerText = "";
        alert("Unable to connect to backend. Please check whether Flask server is running.");
        console.error(error);
    }
}

function applyPredictionColor(elementId, prediction) {
    const element = document.getElementById(elementId);
    element.className = "";

    if (prediction === "Fake") {
        element.classList.add("fake");
    } else {
        element.classList.add("genuine");
    }
}

function applyRiskColor(elementId, riskLevel) {
    const element = document.getElementById(elementId);
    element.className = "";

    if (riskLevel === "High Risk") {
        element.classList.add("high-risk");
    } else if (riskLevel === "Medium Risk") {
        element.classList.add("medium-risk");
    } else {
        element.classList.add("low-risk");
    }
}

function clearResult() {
    document.getElementById("adText").value = "";
    document.getElementById("loadingText").innerText = "";
    document.getElementById("summaryCard").classList.add("hidden");
    document.getElementById("dashboardSection").classList.add("hidden");

    const riskBar = document.getElementById("riskBar");
    if (riskBar) {
        riskBar.style.width = "0%";
    }
}