import React, { useRef } from 'react';

function WebcamSection({ 
  isCameraActive, 
  setIsCameraActive, 
  webcamStream, 
  setWebcamStream, 
  processFile, 
  setActiveTab 
}) {
  const videoRef = useRef(null);

  const startWebcam = async () => {
    try {
      // Stop any existing streams
      if (webcamStream) {
        stopWebcam();
      }

      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setWebcamStream(stream);
      setIsCameraActive(true);
    } catch (err) {
      console.error('Could not access webcam: ' + err.message);
    }
  };

  const stopWebcam = () => {
    if (webcamStream) {
      const tracks = webcamStream.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setWebcamStream(null);
      setIsCameraActive(false);
    }
  };

  const captureImage = () => {
    if (!videoRef.current) return;

    // Create canvas to capture image
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    // Convert to blob
    canvas.toBlob(blob => {
      const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
      processFile(file);
      
      // Switch to upload tab
      setActiveTab('upload');
    }, 'image/jpeg', 0.95);
  };

  return (
    <div className="webcam-section">
      <div className="webcam-container">
        <video 
          ref={videoRef} 
          autoPlay 
          className="webcam-video"
        />
      </div>
      <div className="webcam-controls">
        {!isCameraActive ? (
          <button 
            onClick={startWebcam} 
            className="webcam-button start"
          >
            Start Camera
          </button>
        ) : (
          <>
            <button 
              onClick={captureImage} 
              className="webcam-button capture"
            >
              Capture Image
            </button>
            <button 
              onClick={stopWebcam} 
              className="webcam-button stop"
            >
              Stop Camera
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default WebcamSection;