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
  consult :consultType
}

function SharedConsult({consult} : propsType) {
  const [like, setLike] = useState<boolean>(consult.canLike)
  const {isOpen, onOpen, onOpenChange} = useDisclosure();
  const [openMoreInfo, setOpenMoreInfo] = useState<Boolean>(false)

  useEffect(() => {
    setOpenMoreInfo(false)
  }, [])

  let member = useSelector((state: RootState) => state.member)
  let accessToken = useSelector((state:RootState) => state.accessToken)

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
              <ModalBody>
                <div className="h-[55vh] border-1">
                  여기에 채팅 내용이 들어감
                </div>
                <div className="pl-[68vw] flex-col justifys-center">
                  <p className="text-xs text-center pl-3">공감하기</p>
                  <Button 
                    size='md'
                    className={`
                      ${villageTextColor[member.villageName] || ''} 
                      bg-transparent p-0 pr-1`
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