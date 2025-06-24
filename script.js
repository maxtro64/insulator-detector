document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const resultsDiv = document.getElementById('results');
    const originalImg = document.getElementById('originalImg');
    const annotatedImg = document.getElementById('annotatedImg');
    const detectionsTable = document.getElementById('detectionsTable');

    // Handle drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.background = '#f0f8ff';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.background = '';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.background = '';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            processImage();
        }
    });

    // Handle file selection
    uploadBox.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', processImage);

    async function processImage() {
        const file = fileInput.files[0];
        if (!file) return;

        // Show loading state
        uploadBox.innerHTML = '<p>Processing image...</p>';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                displayResults(data);
            } else {
                throw new Error(data.error || 'Detection failed');
            }
        } catch (error) {
            console.error('Error:', error);
            uploadBox.innerHTML = `
                <div class="upload-prompt">
                    <div class="icon">❌</div>
                    <p>Error: ${error.message}</p>
                </div>
            `;
        }
    }






    function displayResults(data) {
        // Reset upload box
        uploadBox.innerHTML = `
            <div class="upload-prompt">
                <div class="icon">📁</div>
                <p>Drag & drop or click to upload</p>
            </div>
        `;

        // Show results
        resultsDiv.classList.remove('hidden');
        originalImg.src = data.original;
        annotatedImg.src = data.annotated;

        // Populate detections table
        detectionsTable.innerHTML = data.detections.map(det => `
            <tr>
                <td>${det.class}</td>
                <td>${(det.confidence * 100).toFixed(1)}%</td>
                <td>[${det.bbox.join(', ')}]</td>
            </tr>
        `).join('');
    }
});