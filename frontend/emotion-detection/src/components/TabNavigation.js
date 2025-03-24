import React from 'react';

function TabNavigation({ activeTab, setActiveTab }) {
  return (
    <div className="tabs">
      <button 
        className={activeTab === 'upload' ? 'active' : ''}
        onClick={() => setActiveTab('upload')}
      >
        Upload Image
      </button>
      <button 
        className={activeTab === 'webcam' ? 'active' : ''}
        onClick={() => setActiveTab('webcam')}
      >
        Webcam
      </button>
    </div>
  );
}

export default TabNavigation;