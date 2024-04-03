import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../store/store";
import { loadUser } from "../api/member";
import { useDispatch } from "react-redux";
import { memberType } from "../types/DataTypes";
import { saveUserInfo } from "../store/memberSlice";
import TestBackground from "../components/Main/TestBackground";
import { Button, Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure } from "@nextui-org/react";
import { villageBackgroundColor, villageTextColor } from "../atoms/color";
import { changeLang } from "../api/htp";
import { deleteToken, deleteUserInfo } from "../store/memberSlice";
import Swal from "sweetalert2";
import map from './../assets/map.png'
import { useNavigate } from "react-router-dom";


function Main() {
  const dispatch = useDispatch()
  const navigate = useNavigate()

  let member = useSelector((state: RootState) => state.member)

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

  // 로그아웃
  const logout = function() {
    dispatch(deleteToken())
    dispatch(deleteUserInfo())
    Swal.fire({
      text: '로그아웃되었습니다!'
    }).then(() => {
      navigate('/')
    })
  }

  const [display, setDisplay] = useState(true)


  const {isOpen, onOpen, onOpenChange} = useDisclosure();

  return (
    <div className='h-screen relative w-screen'>
      <div className='absolute top-0 left-0 z-0 w-full h-full' onTouchStart={() => setDisplay(false)} onClick={() => setDisplay(false)}>
        <TestBackground />
      </div>
      {/* <div className="absolute w-full z-20">
        <Header />
      </div> */}
      <div className="absolute w-full z-20 flex items-center p-2 justify-between">
        <div className="flex items-center">
          <Button isIconOnly variant="light" onClick={onOpen}>
            <MapIcon />
          </Button>
          <p>지도 보기</p>
        </div>
        <Button 
          isIconOnly 
          variant="light"
          onClick={logout}
          className="ml-1"
        >
          <LogoutIcon />
        </Button>
      </div>
      <div className={`${display ? '' : 'hidden'}`}>
        <div className={`bg-white absolute top-[12%] left-[10%] w-[80%] p-4 text-center text-lg rounded shadow-lg`}>{changeLang(member.villageName)}마을에 오신걸 환영합니다!<br/><span className="text-sm" style={{fontFamily:'JamsilThin'}}>화면을 드래그하면서 둘러보세요</span></div>
        <Button className={`${villageBackgroundColor[member.villageName]} absolute top-[35%] left-[30%]`}>고민상담소</Button>
        <Button className={`${villageBackgroundColor[member.villageName]} absolute bottom-[35%] right-[15%]`}>포스트잇</Button>
        <Button className={`${villageBackgroundColor[member.villageName]} absolute bottom-[10%] right-[20%]`}>마이페이지</Button>
        <Button className={`${villageBackgroundColor[member.villageName]} absolute bottom-[12%] left-[18%]`}>데일리 미션</Button>
      </div>
      <Modal isOpen={isOpen} onOpenChange={onOpenChange}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>🤗 <span className={`${villageTextColor[member.villageName]}`}>{changeLang(member.villageName)}</span>마을에 오신걸 환영해요</ModalHeader>
              <ModalBody>
                <img src={map} alt='지도이미지' className="h-[50vh] w-[70vw] mx-auto"/>
                <p>각 요소들을 누르면 이동할 수 있어요!</p>
              </ModalBody>
              <ModalFooter>
                <Button variant="light" onClick={onClose}>
                  닫기
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
  )
}

export default Main


function MapIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 6.75V15m6-6v8.25m.503 3.498 4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 0 0-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0Z" />
    </svg>

  )
}

function LogoutIcon () {
  return(
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
  <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 9V5.25A2.25 2.25 0 0 1 10.5 3h6a2.25 2.25 0 0 1 2.25 2.25v13.5A2.25 2.25 0 0 1 16.5 21h-6a2.25 2.25 0 0 1-2.25-2.25V15m-3 0-3-3m0 0 3-3m-3 3H15" />
</svg>

  )
}