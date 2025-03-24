import React from 'react';

function SettingsSection({ showBoxes, setShowBoxes }) {
  return (
    <div className="settings">
      <label>
        Show Detection Boxes:
        <input 
          type="checkbox" 
          checked={showBoxes}
          onChange={() => setShowBoxes(!showBoxes)}
        />
      </label>
    </div>
  );
}

export default SettingsSection;