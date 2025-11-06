// =========================================
// SMART STUDY OPTIMIZER - DASHBOARD v4.2
// =========================================

document.addEventListener("DOMContentLoaded", () => {
  console.log("🚀 Dashboard loaded");

  // --- DOM Elements ---
  const startBtn = document.getElementById("start-session-btn");
  const stopBtn = document.getElementById("stop-session-btn");
  const sessionStatus = document.getElementById("session-status");
  const durationDisplay = document.getElementById("session-duration");
  const focusDisplay = document.getElementById("focus-score");
  const recType = document.getElementById("break-type");
  const recMessage = document.getElementById("break-message");
  const recConfidence = document.getElementById("break-confidence");
  const confidenceFill = document.getElementById("confidence-fill");
  const chartCanvas = document.getElementById("focusChart");

  // --- Session UI Containers ---
  const startContainer = document.getElementById("session-start-container");
  const stopContainer = document.getElementById("session-stop-container");
  const timerWrapper = document.getElementById("session-timer-wrapper");
  const metricsGrid = document.getElementById("session-metrics-grid");

  // --- State Variables ---
  let sessionActive = false;
  let timerInterval = null;
  let seconds = 0;
  let focusChart = null;

  // --- Helper: Timer Display ---
  function updateTimer() {
    const mins = Math.floor(seconds / 60);
    durationDisplay.textContent = `${mins}m`;
  }

  // --- Helper: Chart Init ---
  function initChart() {
    if (focusChart) focusChart.destroy();
    focusChart = new Chart(chartCanvas, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Focus Trend",
            data: [],
            borderColor: "#4ade80",
            backgroundColor: "rgba(74, 222, 128, 0.2)",
            borderWidth: 2,
            tension: 0.3,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true, max: 100 } },
        plugins: { legend: { display: false } },
      },
    });
  }

  // --- Helper: Chart Update ---
  async function updateFocusTrend() {
    try {
      const res = await fetch("/api/focus/trend");
      const data = await res.json();
      if (!data.trend) return;
      const labels = data.trend.map((d) => d.time);
      const focusVals = data.trend.map((d) => d.focus);
      focusChart.data.labels = labels;
      focusChart.data.datasets[0].data = focusVals;
      focusChart.update();
    } catch (err) {
      console.error("Trend update failed:", err);
    }
  }

  // --- Fetch AI Recommendation ---
  async function fetchRecommendation() {
    if (!sessionActive) return;
    try {
      const res = await fetch("/api/recommendations/latest");
      const data = await res.json();
      recType.textContent = data.recommendation_type || "ANALYZING";
      recMessage.textContent = data.message || "AI analyzing focus...";
      recConfidence.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
      confidenceFill.style.width = `${Math.round((data.confidence || 0) * 100)}%`;
    } catch (err) {
      console.error("Recommendation fetch failed:", err);
    }
  }

  // --- Fetch Live Session Data ---
  async function updateSessionData() {
    try {
      const res = await fetch("/api/session/current");
      const data = await res.json();
      if (data.active) {
        seconds = data.duration;
        focusDisplay.textContent = `${data.focus_score.toFixed(1)}%`;
        updateTimer();
      }
    } catch (err) {
      console.error("Session fetch failed:", err);
    }
  }

  // --- Start Session ---
  async function startSession() {
    try {
      const res = await fetch("/api/session/start", { method: "POST" });
      const data = await res.json();
      if (data.active) {
        console.log("▶️ Session started:", data);
        sessionActive = true;
        seconds = 0;
        startContainer.style.display = "none";
        stopContainer.style.display = "block";
        timerWrapper.style.display = "flex";
        metricsGrid.style.display = "grid";
        sessionStatus.querySelector("span").textContent = "Active";
        initChart();
        startTimer();
      }
    } catch (err) {
      console.error("Failed to start session:", err);
    }
  }

  // --- Stop Session ---
  async function stopSession() {
    try {
      const res = await fetch("/api/session/stop", { method: "POST" });
      const data = await res.json();
      console.log("⏹ Session stopped:", data);
      sessionActive = false;
      stopTimer();
      startContainer.style.display = "block";
      stopContainer.style.display = "none";
      timerWrapper.style.display = "none";
      metricsGrid.style.display = "none";
      sessionStatus.querySelector("span").textContent = "Inactive";

      alert(`✅ Session saved!\nDuration: ${Math.floor(data.duration / 60)} mins`);
    } catch (err) {
      console.error("Failed to stop session:", err);
    }
  }

  // --- Timer Control ---
  function startTimer() {
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      if (sessionActive) {
        seconds++;
        updateTimer();
        updateSessionData();
        fetchRecommendation();
        updateFocusTrend();
      }
    }, 5000);
  }

  function stopTimer() {
    clearInterval(timerInterval);
  }

  // --- Attach Event Listeners ---
  startBtn.addEventListener("click", startSession);
  stopBtn.addEventListener("click", stopSession);

  // --- Initialize ---
  initChart();
  updateFocusTrend();
});
