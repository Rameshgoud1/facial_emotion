import React from 'react';
import Canvas from './Canvas';
import NoFaceMessage from './NoFaceMessage';

function ImagePreview({ imagePreview, showBoxes, emotions, loading }) {
  if (!imagePreview) return null;

  return (
    <div className="image-container" style={{ position: 'relative', maxWidth: '100%' }}>
      <img 
        id="uploadedImage"
        src={imagePreview} 
        alt="Uploaded" 
        style={{
            width: '700px',    // Fixed width
            height: '400px',   // Fixed height
            objectFit: 'contain', // Ensures the full image is visible without distortion
            display: 'block',
            margin: '0 auto'  // Centers the image
          }}
      />
      {showBoxes && emotions.length > 0 && (
        <Canvas 
          emotions={emotions}
          imagePreview={imagePreview}
        />
      )}
      
      {imagePreview && emotions.length === 0 && !loading && (
        <NoFaceMessage />
      )}
    </div>
  );
}

export default ImagePreview;