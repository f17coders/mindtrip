// MypageHTP.tsx
import { Button } from "@nextui-org/react";
import HTPPicture from "../../../components/MypageHTP/HTPPicture";
import './MypageCss/MypageHTP.css';
import { Link } from "react-router-dom";
import { RootState } from "../../../store/store";
import { useSelector } from "react-redux";
import { villageBackgroundColor } from "../../../atoms/color";


function MypageHTP() {
  let member = useSelector((state:RootState) => state.member)
  return (
    <div className={`${villageBackgroundColor[member.villageName]} flex flex-col justify-around items-center min-h-[80vh] relative`}>
      <div className="bg-white rounded-lg p-4 max-w-L top-32">
        <h1 className="text-lg font-bold mb-4">여기에 검사결과</h1>
        <div>검사결과 박스
          검사결과 어쩌구저쩌구
          잔뜩
          나올
          예정
        </div>
        <Link to="/htp/house">
        <Button className="w-full">새 검사 하러가기</Button>
        </Link>
      </div>
      <div className="gap-4 ml-4 ">
        <HTPPicture title="Home" />
        <HTPPicture title="Tree" />
        <HTPPicture title="Person" />
      </div>
    </div>
  );
}

export default MypageHTP;
