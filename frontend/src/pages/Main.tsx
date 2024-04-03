import { Button } from "@nextui-org/react";
import MainBackground from '../components/Main/MainBackground';
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../store/store";
import { villageBackgroundColor } from "../atoms/color";
import { loadUser } from "../api/member";
import { useDispatch } from "react-redux";
import { memberType } from "../types/DataTypes";
import { saveUserInfo } from "../store/memberSlice";

function Main() {
  const dispatch = useDispatch()
  const [go, setGo] = useState<string|null>(null)

  let member = useSelector((state : RootState) => state.member)

  // 유저 정보 저장
  const saveUser = async function (memberId: number) {
    const userInfo: memberType | void = await loadUser(memberId)
    if (userInfo) {
      dispatch(saveUserInfo(userInfo))
    }
  }

  useEffect(() => {
    if (member.memberId) {
      saveUser(member.memberId)
    }
    setGo(null)
  }, [])

  return (
    <div className='h-screen relative'>
      <div className='absolute top-0 left-0 z-0 w-full h-full'>
        <MainBackground go={go}/>
      </div>
      <div className='relative w-[384px] h-screen mx-auto z-10'>
        <Button
          className={`${go != null ? 'hidden' : ''} ${villageBackgroundColor[member.villageName]} absolute top-[42%] left-[10%]`}
          onClick={() => setGo('consult')}
        >
          고민상담소
        </Button>
        <Button 
          className={`${go != null ? 'hidden' : ''} ${villageBackgroundColor[member.villageName]} absolute bottom-[15%] left-[20%]`}
          onClick={() => setGo('mission')}
        >
          데일리 미션
        </Button>
        <Button 
          className={`${go != null ? 'hidden' : ''} ${villageBackgroundColor[member.villageName]} absolute bottom-[42%] right-[32%]`}
          onClick={() => setGo('postit')}
        >
          포스트잇
        </Button>
        <Button
          className={`${go != null ? 'hidden' : ''} ${villageBackgroundColor[member.villageName]} absolute bottom-[12%] right-[10%]`}
          onClick={() => setGo('mypage')}
        >
          마이페이지
        </Button>
      </div>
    </div>
  )
}

export default Main