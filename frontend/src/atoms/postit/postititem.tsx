// Postititem.tsx
import React from 'react';
import './postit.css';
import PostitLikeBtn from '../buttons/PostitLikeBtn';

type PostitProps = {
  color: string;
  onClick?: () => void; // onClick prop을 선택적으로(Optional) 처리
  children: string;
  style?: React.CSSProperties;
}

const Postit: React.FC<PostitProps> = ({ color, onClick, children, style }) => {
  const postitStyle: React.CSSProperties = {
    backgroundColor: color,
    ...style // 전달받은 style을 포함합니다.
  };

  return (
    <div className="rgyPostIt relative"  onClick={onClick}>
      <PostitLikeBtn/>
      <p>{children}</p>
    </div>
  );
};

export default Postit;
