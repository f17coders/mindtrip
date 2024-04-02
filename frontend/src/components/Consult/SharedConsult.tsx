import { Card, CardBody, Button, Modal, ModalContent, ModalHeader, ModalBody, useDisclosure } from "@nextui-org/react";
import {useEffect, useState} from 'react'
import EmptyHeart from "../../atoms/Icons/EmptyHeart";
import FullHeart from "../../atoms/Icons/FullHeart";
import MoreIcon from "../../atoms/Icons/MoreIcon";
import { villageBackgroundColor, villageTextColor } from "../../atoms/color"
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";
import { consultType } from "../../types/DataTypes";
import axios from "axios";


type propsType = {
  consult :consultType,
  // message: string
}

function SharedConsult({consult} : propsType) {
  const [like, setLike] = useState<boolean>(!consult.canLike)
  const {isOpen, onOpen, onOpenChange} = useDisclosure();
  const [openMoreInfo, setOpenMoreInfo] = useState<Boolean>(false)
  const [recvList, setRecvList] = useState<any[]>([]);
  const [memberId, setMemberId] = useState<number>();

  let member = useSelector((state: RootState) => state.member)
  let accessToken = useSelector((state:RootState) => state.accessToken.value)

  useEffect(() => {
    setOpenMoreInfo(false);

    //공유된 채팅 내역 가져오기
    axios.get(`https://mindtrip.site/api/channels/v1/shared/${consult.channelId}`, {
        headers: {
            Authorization: accessToken
        }
    }).then(response => {
        // console.log(response.data.result);
        setMemberId(response.data.result.memberId)
        setRecvList(response.data.result.messageList);
    }).catch(error => {
        console.error('Error fetching data:', error);
    });
  }, []);



  const handleLike = function() {
    if (like) {
      axios.delete(`https://mindtrip.site/api/consults/v1/like/${consult.consultId}`,{
        headers: {
          Authorization: accessToken
        }
      }) .then(() => setLike(!like))
      .catch((err) => console.log(err))
    } else {
      axios.post(`https://mindtrip.site/api/consults/v1/like/${consult.consultId}`,null, {
        headers: {
          Authorization: accessToken
        }
      }) .then(() => setLike(!like))
      .catch((err) => console.log(err))
    }
  }

  return(
    <div>
      <Card 
        shadow='sm'
        className={`${villageBackgroundColor[member.villageName] || ''} border-1 border-gray-200 h-[20vh] hover:cursor hover:scale-[1.03]`}
      >
        <CardBody onClick={onOpen} className="relative">
          <p className="text-lg">{consult.title}</p>
          <div className="h-14 text-ellipsis overflow-hidden my-1 mb-2">
            <p className="text-sm" style={{fontFamily:"JamsilThin"}}>{consult.content}</p>
          </div>
          <Button 
            size='sm'
            className={`${villageTextColor[member.villageName] || ''} bg-transparent max-w-16 min-w-14 absolute bottom-2 right-2`} 
            startContent={like ? <FullHeart/> : <EmptyHeart/>}
            onClick={handleLike}
          >{like ? consult.likeCount + 1 : consult.likeCount}</Button>
        </CardBody>
      </Card>
      <Modal 
        size="sm" 
        placement='center' 
        isOpen={isOpen} 
        onOpenChange={onOpenChange}
      >
        <ModalContent>
          {() => (
            <>
              <ModalHeader>
                <div className="flex items-center relactive">
                  <p className="mr-1">{consult.title}</p>
                  <Button 
                    isIconOnly
                    onClick={() => {setOpenMoreInfo(!openMoreInfo)}}
                    className={`
                      ${villageTextColor[member.villageName] || ''} 
                      bg-transparent p-0 pr-1`
                    }
                  >
                    <MoreIcon/>
                  </Button>
                  <div 
                    className={
                      `absolute ${openMoreInfo ? '' : 'hidden'}
                      w-[80vw] h-[20vh] top-[9vh] 
                      bg-white p-3 text-sm
                      rounded-md shadow-xl 
                      ${villageBackgroundColor[member.villageName]}
                      `
                    }
                    style={{fontFamily:'JamsilThin'}}
                  >
                    {consult.content}
                  </div>
                </div>
              </ModalHeader>
              <ModalBody className="relactive">
                <div className="h-[55vh] border-1 overflow-y-scroll p-2 mb-[10vh]">
                  {/* recvList가 비어 있는지 여부를 체크하여 처리 */}
                  {recvList.length > 0 && recvList.map((msg, index) => (
                    <div key={index}>
                      {msg.sender.memberId == memberId ? (
                        <MyBallon message={msg.text} />
                      ) : (
                        <OtherBallon message={msg.text} />
                      )}
                    </div>
                  ))}
                </div>
                <div className="absolute bottom-[2vh] right-[2vw] flex-col justifys-center">
                  <p className="text-xs text-center text-gray-400">공감하기</p>
                  <Button 
                    size='md'
                    className={`
                      ${villageTextColor[member.villageName] || ''} 
                      bg-transparent p-0 pr-1 h-[5vh]`
                    } 
                    startContent={like ? <FullHeart/> : <EmptyHeart/>}
                    onClick={() => setLike(!like)}
                  >{like ? 21 : 20}</Button>
                </div>
              </ModalBody>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
    
  )
}

export default SharedConsult


function OtherBallon({ message }: { message: string }) {
  return (
    <div className={`bg-gray-300 rounded-lg w-fit max-w-[90%] py-1 px-2 my-2`}>
      {message}
    </div>
  )
}

function MyBallon({ message }: { message: string }) {
  let member = useSelector((state: RootState) => state.member)
  return (
    <div className='flex justify-end'>
      <div className={`${villageBackgroundColor[member.villageName]} rounded-lg w-fit max-w-[90%] py-1 px-2 my-2 mr-1.5`}>
        {message}
      </div>
    </div>
  )
}