import React, { useRef } from 'react';

function UploadSection({ processFile }) {
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  };

  return (
    <div className="upload-section">
      <input 
        type="file" 
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept="image/*"
        style={{display: 'none'}}
      />
      <button 
        onClick={() => fileInputRef.current.click()}
        className="upload-button"
      >
        Choose Image
      </button>
    </div>
  );
}

export default UploadSection;