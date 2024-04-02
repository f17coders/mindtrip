// import React, { useState } from 'react';
import MyPostit from "../../../components/MyPostit/MyPostit";

// interface Memo {
//   date: Date;
//   content: string;
// }

// interface Props {
//   memos: Memo[]; // Memo 배열을 props로 받음
// }

function MypageHistory() {
  return (
    <div className="bg-[#f4c2c2]">
      <div className="flex justify-center items-center w-[20vw] mx-auto h-[80vh]">
        <MyPostit />
    </div>
    </div>
  );
}

export default MypageHistory;
