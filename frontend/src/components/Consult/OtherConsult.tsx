import { Card, CardBody, Button, Modal, ModalContent, ModalHeader, ModalBody, useDisclosure } from "@nextui-org/react";
import { toggleOpen, changeList, changeSelectedId, setisMine } from "./../../store/chatSlice";
import { useSelector, useDispatch } from "react-redux";
import { RootState } from './../../store/store'
import { consultType } from "../../types/DataTypes";
import { villageBackgroundColor, villageTextColor } from "../../atoms/color";
import { enterRoom } from "../../api/consults";
// 다른 사람의 고민 하나의 컴포넌트
// 고민 제목과 내용, 그리고 상담하기 버튼이 들어감
// 상담하기 버튼을 누르면 전체 내용, 참여 여부를 확인하는 모달이 뜨고
// 참여하기를 누르면 바로 채팅방으로 넘어가야함


// 모달 제어용 타입 지정
type useDisclosureType = {
  isOpen: boolean
  onOpen: () => void
  onOpenChange: (isOpen: boolean) => void
}

// 고민 타입
type propsType = {
  consult :consultType
}

function OtherConsult({consult} : propsType) {
  // 모달창 오픈 제어용
  const { isOpen, onOpen, onOpenChange }: useDisclosureType = useDisclosure();

  // 채팅창 관련 가져오기
  let chat = useSelector((state:RootState)=> state.chat)
  const dispatch = useDispatch()

  let member = useSelector((state:RootState)=> state.member)
  let accessToken = useSelector((state:RootState) => state.accessToken.value)
  const handleEnter = async (consultId:number) => {
    try {
      const tempChannelId = await enterRoom(accessToken, consultId)
      if (typeof tempChannelId == 'string') {
        // 채널 아이디 알려주고
        dispatch(changeSelectedId(tempChannelId))
        dispatch(setisMine(false))
        // 만약 채팅창이 닫혀있는 상태라면 열어주고
        if (chat.isOpen === false) {
          dispatch(toggleOpen())
        } 
        // 채팅방으로 옮겨주자
        dispatch(changeList(false))
      }
    } catch (err) {
      console.log(err)
    }
  }

  // 고민방 참여
  const enterChat = function() {
    // 채널에 참여하는걸 알려주고
    if (consult.consultId != null) {
      handleEnter(consult.consultId)
    }
    
  }

  return (
    <Card 
      shadow='sm'
      className={`${villageBackgroundColor[member.villageName]} h-full`}
    >
      <CardBody className="p-2">
        <p className="text-md py-2 h-[9vh] font-bold text-center">{consult.title}</p>
        <Button 
          variant="light" 
          onPress={onOpen}
          className={`mx-auto mt-2 max-w-40 min-w-32 ${villageTextColor[member.villageName]} bg-white`}
        >자세히 보기</Button>
      </CardBody>
      <Modal size="sm" placement='center' isOpen={isOpen} onOpenChange={onOpenChange}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader className="text-sm text-gray-500" style={{fontFamily:'JamsilThin'}}>{consult.nickname}님의 고민</ModalHeader>
              <ModalBody className="flex-col items-center">
                <div className="min-h-[25vh] text-center">
                  <p className="text-xl mb-3">{consult.title}</p>
                  <p className="min-h-[20vh]" style={{fontFamily:'JamsilThin'}}>{consult.content}</p>
                </div>
                <Button 
                  className={`w-40 ${villageBackgroundColor[member.villageName]} mb-[3vh]`} 
                  onClick={() => {
                    onClose()
                    enterChat()
                  }}
                >고민 들어주기</Button>
              </ModalBody>
            </>
          )}
        </ModalContent>
      </Modal>
    </Card>
  )
}

export default OtherConsult