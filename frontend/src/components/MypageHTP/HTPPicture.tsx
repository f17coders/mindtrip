// HTPPicture.tsx
import HTPDrawing from './HTPDrawing';
import { pictureResultType } from '../../types/DataTypes';

interface HTPPictureProps {
  data: pictureResultType;
}

function HTPPicture({ data }:HTPPictureProps) {
  return (
    <div className='w-full'>
      <div className='mt-6'>{data.test_time.toString()}</div>
      <div className="max-w-[90vw] flex overflow-x-scroll">
        <HTPDrawing imageUrl={data.house_url} />
        <HTPDrawing imageUrl={data.tree_url} />
        <HTPDrawing imageUrl={data.person_url} />
      </div>
    </div>
  );
};

export default HTPPicture;

