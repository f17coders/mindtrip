// SuccessButton 컴포넌트

import { Button } from "@nextui-org/react";

interface SuccessButtonProps {
  isFinish: boolean;
  missionId: number; // 미션 ID를 받아옴
  onClick: (missionId: number) => void; // onClick 이벤트를 필수로 받도록 수정
}

function SuccessButton({ isFinish, missionId, onClick }: SuccessButtonProps) {
  // isFinish 값을 초기 상태로 설정
  
  // 버튼 클릭 시 호출되는 함수
  const handleButtonClick = async () => {
    onClick(missionId)
  };

  return (
    <Button
      radius="full"
      className={`bg-gradient-to-tr from-pink-500 to-yellow-500 text-white shadow-lg ${
        isFinish ? "cursor-not-allowed opacity-50" : ""
      }`}
      onClick={handleButtonClick}
      disabled={isFinish}
    >
      {isFinish ? "미션 완료" : "미션 성공"}
    </Button>
  );
}

export default SuccessButton;
