// HTPDrawing.tsx
import React from 'react';

interface HTPDrawingProps {
  imageUrl: string;
}

const HTPDrawing: React.FC<HTPDrawingProps> = ({ imageUrl }) => {
  return (
    <div className="mycard">
      <div className="bg">
        <img src={imageUrl} alt="Drawing" className='w-full h-full' />
      </div>
    </div>
  );
}

export default HTPDrawing;

