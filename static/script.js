document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileFeedback = document.getElementById('fileFeedback');
    const uploadForm = document.getElementById('uploadForm');

    // Trigger file chooser layout allocation
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileFeedback.textContent = `Selected: ${fileInput.files[0].name}`;
            fileFeedback.style.display = 'block';
        }
    });

    // Extract numerical dashboard entries via backend service pipeline
    async function fetchAnalytics() {
        try {
            const response = await fetch('/api/analytics');
            const data = await response.json();
            
            if (data.total_scans_processed) {
                document.getElementById('statTotalScans').textContent = data.total_scans_processed;
                document.getElementById('statAvgConf').textContent = data.average_system_confidence;
            }
        } catch (err) {
            console.error("Historical dataset evaluation error:", err);
        }
    }

    // Submit image stream pipeline
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert("Please select an image file first.");
            return;
        }

        // Standard toggle display states
        document.getElementById('loader').classList.remove('hidden');
        document.getElementById('resultsCard').classList.add('hidden');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('username', document.getElementById('username').value || "Guest");

        try {
            const response = await fetch('/api/predict', { method: 'POST', body: formData });
            const result = await response.json();
            
            document.getElementById('loader').classList.add('hidden');
            
            if (result.status === 'success') {
                document.getElementById('resPlant').textContent = result.plant;
                document.getElementById('resDisease').textContent = result.disease;
                document.getElementById('resConfidence').textContent = `Confidence: ${result.confidence}`;
                document.getElementById('resSymptoms').textContent = result.symptoms;
                document.getElementById('resRemedy').textContent = result.remedy;
                
                document.getElementById('resultsCard').classList.remove('hidden');
                
                // Refresh core dashboard analytics views instantly
                fetchAnalytics();
            } else {
                alert(`Error processing data inputs: ${result.detail || "Verification failure"}`);
            }
        } catch (err) {
            document.getElementById('loader').classList.add('hidden');
            alert("Network error. Verify server is running.");
        }
    });

    // Run baseline check on structural layout initialization
    fetchAnalytics();
});
