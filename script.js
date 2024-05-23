function procceed() {
  window.location.href = "index.html";
}
function chooseText() {
  window.location.href = "text.html";
}

function chooseImage() {
  window.location.href = "image.html";
}

function analyzeText() {
  var text = document.getElementById("textInput").value;

  // Send text to backend for analysis
  fetch('/classify-text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text: text })
  })
  .then(response => response.json())
  .then(data => {
    // Display result
    document.getElementById("textResult").innerText = "Text classified as: " + data.prediction;
  
    // Edit input style for the image to fit
    document.getElementById("textInput").classList.add('analysedTextInput');
  
    // Display image
    if (data.prediction === "AI-generated") {
      document.getElementById("textResultImage").innerHTML = "<img src='imgs/AI_detected.png' class='resultimgs' alt='AI Generated Image'>";
      document.getElementById("textResult").classList.add('redText');
    } else {
      document.getElementById("textResultImage").innerHTML = "<img src='imgs/HUMAN_detected.png'  class='resultimgs' alt='Human Written Image'>";
      document.getElementById("textResult").classList.add('greenText');
    }
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

function analyzeImage() {
  var fileInput = document.getElementById("imageInput");
  var file = fileInput.files[0];
  
  // Check if a file is selected
  if (file) {
    // Hide the "Select an image please" text
    document.getElementById("imgSelectText").style.display = "none";
    document.getElementById("fileNameDisplay").style.display = "none";
  
    // Send image to backend for analysis
    var formData = new FormData();
    formData.append('image', file);
  
    fetch('/classify-image', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      // Display result
      document.getElementById("imageResult").innerText = "Image classified as: " + data.prediction;
  
      // Edit input style for the image to fit
      document.getElementById("imageInputLabel").classList.add('analysedImageInput');
  
      // Display image
      if (data.prediction === "AI-generated") {
        document.getElementById("imageResultImage").innerHTML = "<img src='imgs/AI_detected.png' class='resultimgs' alt='AI Generated Image'>";
        document.getElementById("imageResult").classList.add('redText');//change result color
      } else {
        document.getElementById("imageResultImage").innerHTML = "<img src='imgs/HUMAN_detected.png' class='resultimgs' alt='Human Written Image'>";
        document.getElementById("imageResult").classList.add('greenText');//change result color
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
  } 
}
