import { useEffect } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../store/store";
import { loadUser } from "../api/member";
import { useDispatch } from "react-redux";
import { memberType } from "../types/DataTypes";
import { saveUserInfo } from "../store/memberSlice";
import Header from "../components/Header";
import TestBackground from "../components/Main/TestBackground";

function Main() {
  const dispatch = useDispatch()

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
  }, [])

  return (
    <div className='h-screen relative'>
      <div className='absolute top-0 left-0 z-0 w-full h-full z-0'>
        <TestBackground />
      </div>
      <div className="absolute w-full z-20">
        <Header />
      </div>
    </div>
  )
}

export default Main