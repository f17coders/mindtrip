// MypageHTP.tsx
import { Button } from "@nextui-org/react";
import HTPPicture from "../../../components/MypageHTP/HTPPicture";
import './MypageCss/MypageHTP.css';
import { Link } from "react-router-dom";
import { RootState } from "../../../store/store";
import { useSelector } from "react-redux";
import { villageBackgroundColor } from "../../../atoms/color";
import { loadRecentResult, loadPictureResult } from "../../../api/htp";
import { useEffect, useState } from "react";
import { pictureResultType } from "../../../types/DataTypes";

function MypageHTP() {
  let member = useSelector((state:RootState) => state.member)
  let accessToken = useSelector((state:RootState) => state.accessToken.value)
  const [result1, setResult1] = useState<string>('')
  const [result2, setResult2] = useState<pictureResultType[]>([])

  // 문장결과 가져오기
  const getResult1 = async () => {
    const tempresult = await loadRecentResult(accessToken)
    if (typeof tempresult === 'string') {
      setResult1(tempresult)
    }
  }

  // 사진 결과들 가져오기
  const getResult2 = async () => {
    const tempResult = await loadPictureResult(accessToken)
    if (tempResult) {
      setResult2(tempResult)
    }
  }
  
  useEffect(()=> {
    getResult1()
    getResult2()
  }, [])

  return (
    <div className={`${villageBackgroundColor[member.villageName]} w-screen flex flex-col justify-around items-center min-h-[80vh]`}>
      <div className="bg-white rounded-lg p-4 mt-5 w-[90vw]">
        <h1 className="text-lg font-bold mb-2">최근 HTP 검사 결과</h1>
        <div className="py-3" style={{fontFamily:"JamsilThin"}}>{result1}</div>
        <Link to="/htp/house">
          <Button className="w-full">새 검사 하러가기</Button>
        </Link>
      </div>
      <div className="grid gap-y-4">
        {
          result2.map((res, idx) => {
            return (
              <div key={idx}>
                <HTPPicture data={res} />
              </div>
            )
          })
        }
        
      </div>
    </div>
  );
}

export default MypageHTP;
