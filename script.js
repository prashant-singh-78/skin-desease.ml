document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const resultContainer = document.getElementById('result-container');
    const selectedImage = document.getElementById('selected-image');
    const resetBtn = document.getElementById('reset-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    
    const predictedClass = document.getElementById('predicted-class');
    const mainConfidence = document.getElementById('main-confidence');
    const secondaryResults = document.getElementById('secondary-results');

    let currentFile = null;

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('highlight'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                selectedImage.src = e.target.result;
                dropZone.classList.add('hidden');
                previewContainer.classList.remove('hidden');
                resultContainer.classList.add('hidden');
            }
            reader.readAsDataURL(currentFile);
        }
    }

    resetBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Show result container and loader
        resultContainer.classList.remove('hidden');
        loader.classList.remove('hidden');
        resultContent.classList.add('hidden');
        
        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth' });

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                loader.classList.add('hidden');
                return;
            }

            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Something went wrong. Make sure the backend server (main.py) is running.');
            loader.classList.add('hidden');
        }
    });

    function displayResults(data) {
        loader.classList.add('hidden');
        resultContent.classList.remove('hidden');

        predictedClass.textContent = data.prediction;
        mainConfidence.textContent = `${data.confidence.toFixed(1)}% Confidence`;

        // Clear previous secondary results
        secondaryResults.innerHTML = '';

        // Add all predictions as bars
        data.all_predictions.forEach(res => {
            const barHtml = `
                <div class="prob-bar-container">
                    <div class="prob-header">
                        <span>${res.class}</span>
                        <span>${res.confidence.toFixed(1)}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: 0%"></div>
                    </div>
                </div>
            `;
            secondaryResults.insertAdjacentHTML('beforeend', barHtml);
        });

        // Animate bars
        setTimeout(() => {
            const fills = secondaryResults.querySelectorAll('.bar-fill');
            data.all_predictions.forEach((res, i) => {
                fills[i].style.width = `${res.confidence}%`;
            });
        }, 100);
    }
});
