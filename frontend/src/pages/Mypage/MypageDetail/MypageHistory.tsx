// import React, { useState } from 'react';
import MyPostit from "../../../components/MyPostit/MyPostit";
import { RootState } from "../../../store/store";
import { useSelector } from "react-redux";
import { villageBackgroundColor } from "../../../atoms/color";



// interface Memo {
  //   date: Date;
  //   content: string;
  // }
  
  // interface Props {
    //   memos: Memo[]; // Memo 배열을 props로 받음
    // }
    
    function MypageHistory() {
  let member = useSelector((state:RootState) => state.member)
  return (
    <div className={`${villageBackgroundColor[member.villageName]} flex flex-col justify-around items-center min-h-[80vh] relative`}>
      <div className="flex justify-center items-center w-[20vw] mx-auto h-[80vh]">
      <div>
        <MyPostit />
      </div>
    </div>
    </div>
  );
}

export default MypageHistory;
