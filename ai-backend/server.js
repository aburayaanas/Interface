const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve home.html as the landing page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'home.html'));
});

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Set up storage for multer
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  }
});
const upload = multer({ storage: storage });

// Endpoint to handle text input
app.post('/analyze-text', async (req, res) => {
  const { text } = req.body;
  try {
    const response = await axios.post('http://localhost:5000/predict', { text });
    res.json(response.data);
  } catch (error) {
    res.status(500).send('Error analyzing text');
  }
});

// Endpoint to handle image input
app.post('/analyze-image', upload.single('image'), async (req, res) => {
  const imagePath = req.file.path;
  try {
    const response = await axios.post('http://localhost:5000/predict-image', { imagePath });
    res.json(response.data);
  } catch (error) {
    res.status(500).send('Error analyzing image');
  } finally {
    fs.unlinkSync(imagePath); // Clean up the uploaded file
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
