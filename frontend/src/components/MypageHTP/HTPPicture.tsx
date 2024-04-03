// HTPPicture.tsx
import HTPDrawing from './HTPDrawing';
import { pictureResultType } from '../../types/DataTypes';

interface HTPPictureProps {
  data: pictureResultType;
}

function HTPPicture({ data }:HTPPictureProps) {
  const date = new Date(data.test_time)
  const year = date.getFullYear()
  const month = date.getMonth()
  const day = date.getDate()
  const formattedDate = `${year}년 ${month + 1}월 ${day}일의 검사`

  return (
    <div className='w-full'>
      <div className='my-4'>{formattedDate}</div>
      <div className="max-w-[90vw] flex overflow-x-scroll">
        <HTPDrawing imageUrl={data.house_url} />
        <HTPDrawing imageUrl={data.tree_url} />
        <HTPDrawing imageUrl={data.person_url} />
      </div>
    </div>
  );
};

export default HTPPicture;

