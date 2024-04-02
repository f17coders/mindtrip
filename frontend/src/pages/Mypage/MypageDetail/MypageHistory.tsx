// import React, { useState } from 'react';
import MyPostit from "../../../components/MyPostit/MyPostit";
import { useSelector } from "react-redux";
import { RootState } from "../../../store/store";
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
    <div className={`${villageBackgroundColor[member.villageName]} h-[80vh] w-screen`}>
        <MyPostit />
    </div>
  );
}

export default MypageHistory;
