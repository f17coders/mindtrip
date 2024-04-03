import { useState, useEffect } from 'react'
import { Button, Select, SelectItem, Tooltip, Card, Modal, ModalContent, ModalHeader, ModalBody, useDisclosure } from "@nextui-org/react";
import OtherConsult from '../../components/Consult/OtherConsult';
import { useSelector, useDispatch } from "react-redux";
import { RootState } from './../../store/store'
import { toggleOpen } from '../../store/chatSlice';
import ChatIcon from './../../atoms/Icons/ChatIcon'
import XIcon from '../../atoms/Icons/XIcon';
import PencilIcon from '../../atoms/Icons/PencilIcon';
import Chat from '../../components/Consult/Chat/Chat';
import { consultType, categoryType } from '../../types/DataTypes';
import { getConsults } from '../../api/consults';
import CreateNewConsult from '../../components/Consult/CreateNewConsult';
import { villageBackgroundColor, villageTextColor } from '../../atoms/color';
import Header from '../../components/Header';
import axios from 'axios';

// 모달 제어용 타입 지정
type useDisclosureType = {
  isOpen: boolean
  onOpen: () => void
  onOpenChange: (isOpen: boolean) => void
}

function ConsultOther() {
  const dispatch = useDispatch()

  // 카테고리 받기
  let category = useSelector((state: RootState) => state.consultSlice.category)

  // 멤버정보
  let member = useSelector((state: RootState) => state.member)
  let accessToken = useSelector((state:RootState) => state.accessToken.value)


   // 고민 받아서 저장할 변수
  const [otherConsults, setOtherConsult] = useState<consultType[]>([])
  
  
  // 선택된 카테고리
  const [selectedCategory, setSelectedCategory] = useState<categoryType | null>(null)
  const handleCategory = (e: any) => {
    setSelectedCategory(e.target.value)
    console.log(selectedCategory)
    axios.get(`https://mindtrip.site/api/consults/v1/category/${e.target.value}`,{
      headers: {
        Authorization: accessToken
      }
    }).then((res) => {
      setOtherConsult(res.data.result.consultList)
    }) .catch((err) => console.log(err))
  }
 

  useEffect(() => {
    // 전체 고민 가져오기
    const fetchConsult = async () => {
      try {
        let tempOtherConsult: consultType[] = await getConsults(accessToken)
        setOtherConsult(tempOtherConsult)
        console.log(tempOtherConsult)
      } catch (err) {
        console.log(err)
      }
    }
    fetchConsult()
  }, [])

  // 채팅 관련 가져오기
  let chat = useSelector((state: RootState) => state.chat)

  // 모달창 오픈 제어용
  const { isOpen, onOpen, onOpenChange }: useDisclosureType = useDisclosure();

  return (
    <div className="w-full lg:w-3/4 mx-auto h-screen">
      <Header />
      <div className="px-3 min-h-[40%]">
        <p className="text-2xl hover:cursor-pointer mb-3">🙋‍♀️다른 사람들의 고민 보기</p>
        <div className="flex justify-between items-center mt-4 mb-2">
          {/* 카테고리들 */}
          <Select
            label='카테고리 선택'
            size='sm'
            onChange={handleCategory}
            className='w-[150px]'
            style={{fontFamily:'JamsilThin'}}
          >
            {category.map((oneCategory: categoryType) => {
              return (
                <SelectItem key={oneCategory.categoryId} style={{fontFamily:'JamsilThin'}}>
                  {oneCategory.categoryName}
                </SelectItem>
              )
            })
            }
          </Select>

          {/* 내 고민 작성하기 */}
          <Tooltip content='내 고민 작성하기'>
            <Button isIconOnly variant='light' onPress={onOpen}><PencilIcon /></Button>
          </Tooltip>
        </div>
        <div className='grid grid-cols-2'>
        {
          otherConsults?.map((consult, idx) => (
            <div className="w-full h-[20vh] p-2" key={idx}>
              {consult.isClosed === false && <OtherConsult consult={consult} />}
            </div>
          ))
        }
        {
          otherConsults?.length === 0 ? (
            <div className='h-[20vh] text-gray-400 text-sm' style={{fontFamily:'JamsilThin'}}>
              아직 업로드된 고민이 없습니다!
            </div>
          ) : null
        }
        </div>
      </div>
      {/* 채팅 창 여는 버튼 */}
      <Tooltip content={chat.isOpen ? '닫기' : '대화 확인하기'} placement={chat.isOpen ? 'right' : 'top'}>
        <Button
          isIconOnly
          size='lg'
          radius='full'
          variant={chat.isOpen ? 'solid' : 'flat'}
          onClick={() => dispatch(toggleOpen())}
          className={`${villageBackgroundColor[member.villageName]} ${villageTextColor[member.villageName]} fixed bottom-[3%] right-[4%] shadow-xl border-1 border-zinc-400 shadow`}
        >
          {chat.isOpen ? <XIcon /> : <ChatIcon />}
        </Button>
      </Tooltip>
      {/* 채팅창 */}
      <Card
        style={{
          display: chat.isOpen ? 'block' : 'none',
        }}
        className='fixed top-[20%] right-[2.5%] w-[80%] h-[70%]
          sm:top-[20%] w-[95%] h-[65%] p-5 z-10'
      >
        <Chat />
      </Card>
      <Modal size="sm" placement='center' isOpen={isOpen} onOpenChange={onOpenChange}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>내 고민 공유하기</ModalHeader>
              <ModalBody>
                <CreateNewConsult onClose={onClose} category={category} />
              </ModalBody>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
  )
}

export default ConsultOther