import React, { useState} from 'react';
import './App.css';
import Header from './components/Header';
import TabNavigation from './components/TabNavigation';
import UploadSection from './components/UploadSection';
import WebcamSection from './components/WebcamSection';
import SettingsSection from './components/SettingsSection';
import ImagePreview from './components/ImagePreview';
import ResultsSection from './components/ResultsSection';

function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [emotions, setEmotions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [activeTab, setActiveTab] = useState('upload');
  
  // Webcam states
  const [webcamStream, setWebcamStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  const API_URL = 'http://localhost:5000';

  const processFile = (file) => {
    // Reset previous state
    setImagePreview(null);
    setEmotions([]);
    setError(null);

    // Read file
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target.result);
      uploadImage(file);
    };
    reader.readAsDataURL(file);
  };

  const uploadImage = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Image upload failed');
      }

      const data = await response.json();
      setEmotions(data.emotions || []);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      
      <TabNavigation 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
      />

      {activeTab === 'upload' && (
        <UploadSection 
          processFile={processFile} 
        />
      )}

      {activeTab === 'webcam' && (
        <WebcamSection 
          isCameraActive={isCameraActive}
          setIsCameraActive={setIsCameraActive}
          webcamStream={webcamStream}
          setWebcamStream={setWebcamStream}
          processFile={processFile}
          setActiveTab={setActiveTab}
        />
      )}

      <SettingsSection 
        showBoxes={showBoxes}
        setShowBoxes={setShowBoxes}
      />

      {loading && <div className="loading">Processing...</div>}
      {error && <div className="error">{error}</div>}

      <ImagePreview 
        imagePreview={imagePreview}
        showBoxes={showBoxes}
        emotions={emotions}
        loading={loading}
      />

      {emotions.length > 0 && (
        <ResultsSection emotions={emotions} />
      )}
    </div>
  );
}

export default App;
