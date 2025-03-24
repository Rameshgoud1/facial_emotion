import React from 'react';

function NoFaceMessage() {
  return (
    <div className="no-face-found" style={{
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      backgroundColor: 'rgba(0,0,0,0.7)',
      color: 'white',
      padding: '20px',
      borderRadius: '10px',
      fontSize: '24px',
      textAlign: 'center',
      zIndex: 10
    }}>
      No Face Detected
    </div>
  );
}

export default NoFaceMessage;