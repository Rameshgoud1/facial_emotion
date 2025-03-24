import React, { useRef, useEffect } from 'react';

function Canvas({ emotions, imagePreview }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const drawBoundingBoxes = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        console.error("Canvas element not found!");
        return;
      }
  
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        console.error("Could not get 2D context!");
        return;
      }
  
      const image = document.getElementById("uploadedImage");
      if (!image) {
        console.error("Image element not found!");
        return;
      }
  
      // Set canvas size
      canvas.width = image.width;
      canvas.height = image.height;
  
      // Precise scale factors
      const scaleX = image.width / image.naturalWidth;
      const scaleY = image.height / image.naturalHeight;
  
      console.log(`Precise ScaleX: ${scaleX}, ScaleY: ${scaleY}`);
  
      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);
  
      // Draw boxes
      emotions.forEach((emotion) => {
        const [x1, y1, x2, y2] = emotion.bbox;
        
        // Fine-tune the bounding box to more precisely fit the face
        const faceWidth = x2 - x1;
        const faceHeight = y2 - y1;
        
        // Adjust box size
        const paddingX = faceWidth * 0.25;  // 20% padding on each side
        const paddingY = faceHeight * 0.10;  // 10% padding on top and bottom
        
        const adjustedX1 = x1 + paddingX;
        const adjustedY1 = y1 + paddingY;
        const adjustedX2 = x2 - paddingX;
        const adjustedY2 = y2 - paddingY;
        
        const scaledX = adjustedX1 * scaleX;
        const scaledY = adjustedY1 * scaleY;
        const scaledWidth = (adjustedX2 - adjustedX1) * scaleX;
        const scaledHeight = (adjustedY2 - adjustedY1) * scaleY;
  
        console.log(`Drawing adjusted box: 
          x=${scaledX}, 
          y=${scaledY}, 
          width=${scaledWidth}, 
          height=${scaledHeight}`);
  
        // Draw rectangle
        ctx.beginPath();
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.rect(scaledX, scaledY, scaledWidth, scaledHeight);
        ctx.stroke();
        
        ctx.fillStyle = 'green'; // Background color
        ctx.fillRect(scaledX, scaledY - 30, 120, 25); // Background rectangle
        
        ctx.fillStyle = 'white'; // Text color
        ctx.font = '20px Arial'; // Font size
        ctx.fillText(
          `${emotion.emotion} (${(emotion.confidence * 100).toFixed(0)}%)`, 
          scaledX, 
          scaledY - 10
        );
      });
    };
  
    // Ensure image is loaded and emotions are detected
    if (imagePreview && emotions.length > 0) {
      // Use multiple delays to ensure rendering
      const timeouts = [
        setTimeout(drawBoundingBoxes, 100),
        setTimeout(drawBoundingBoxes, 500),
        setTimeout(drawBoundingBoxes, 1000)
      ];
  
      window.addEventListener("resize", drawBoundingBoxes);
  
      return () => {
        // Clear all timeouts
        timeouts.forEach(clearTimeout);
        window.removeEventListener("resize", drawBoundingBoxes);
      };
    }
  }, [imagePreview, emotions]);

  return (
    <canvas 
      ref={canvasRef} 
      id="faceCanvas"
      style={{ 
        position: 'absolute', 
        top: 0, 
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none'
      }}
    />
  );
}

export default Canvas;