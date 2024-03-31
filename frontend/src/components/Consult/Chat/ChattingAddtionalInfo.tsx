import { Button } from "@nextui-org/react"
import Swal from "sweetalert2"
import { useDispatch } from "react-redux"
import { toggleOpen, changeList } from "../../../store/chatSlice"
import { useEffect, useState } from "react"
import axios from 'axios'
import { useSelector } from "react-redux"
import { RootState } from "../../../store/store"

type propsType = {
  isMine :Boolean
}

type chatInfo = {
  title: string,
  content: string,
  closed: boolean,
  shared: boolean|null,
  consultId: string|null
}

function ChattingAdditionalInfo({isMine} : propsType) {
  const dispatch = useDispatch()

  let chat = useSelector((state : RootState) => state.chat)
  let accessToken = useSelector((state:RootState) => state.accessToken)

  const [chatInfo, setChatInfo] = useState<chatInfo|null>(null)
  const loadChatInfo = async () => {
    try {
      const res = await axios.get(`https://mindtrip.site/api/consults/v1/detail/${chat.selectedId}`, {
        headers: {
          Authorization: accessToken
        }
      })
      setChatInfo(res.data.result)
    } catch(err) {
      console.log(err)
    }
  }

  useEffect(() => {
    loadChatInfo()
  }, [])

  const endConsult = function () {
    Swal.fire({
      title: '고민을 종료합니다',
      text: '이 대화를 공유할까요?'
    }).then((result) => {
      // 여기에 종료 로직
      axios.put(`https://mindtrip.site/api/consults/v1/close/${chatInfo?.consultId}`, {
        'isShared' : result.isConfirmed ? true : false
      }, {
        headers : {
          Authorization: accessToken
        }
      }).then(() => {
        // 창 닫아줌
        dispatch(toggleOpen())
        // 다음에 리스트로 가게
        dispatch(changeList(true))
      }).catch((err) => console.log(err))
    })
  }

  // 사용자 내보내는 함수
  const leaveChat = function () {
    Swal.fire({
      title: '현재 상담에서 나갑니다'
    }).then(() => {
      axios.put(`https://mindtrip.site/api/consults/v1/exit/${chatInfo?.consultId}`, {
        headers : {
          Authorization: accessToken
        }
      }).then(() => {
        // 리스트로 가게해줌
        dispatch(changeList(true))
      }).catch((err) => console.log(err))
    })
  }

  // 내보내는 함수
  const kickUser = function () {
    Swal.fire({
      title: '현재 채팅중인 대상을 내보냅니다'
    }).then(() => {
      // 여기에 내보내는 로직
      axios.put(`https://mindtrip.site/api/channels/v1/expel/${chatInfo?.consultId}`,{
        headers:{
          Authorization: accessToken
        }
      }).then(() => {
        Swal.fire({text:'내보내기 성공!'})
        dispatch(changeList(true))
      }).catch((err) => console.log(err))


      Swal.fire({
        title: '사용자를 내보냈습니다'
      }).then(() => {
        // 리스트로 가게해줌
        dispatch(changeList(true))
      })
    })
  }

  useEffect(() => {
    console.log(isMine)
  } ,[])

  return (
    <div
      className="absolute top-[15%] border-1 rounded-xl p-3 w-[90%] h-[40%] bg-white"
    >
      <p>고민제목</p>
      <div className="min-h-[60%] overflow-scroll">
        <p className="text-sm">고민내용</p>
      </div>
      {/* 내가 만든방이라면 */}
      {
        isMine === true && (<div className="flex justify-end">
          <Button
            size="sm"
            variant="ghost"
            color="danger"
            onClick={kickUser}
          >
            상대 내보내기
          </Button>
          <Button
            size="sm"
            variant="ghost"
            color="primary"
            onClick={endConsult}
            className="ml-3"
          >
            고민 종료하기
          </Button>
        </div>)
      }
      {/* 내가만든 방이 아니라면 */}
      {
        isMine === false && (<div className="flex justify-end">
          <Button
            size="sm"
            variant="ghost"
            color="danger"
            onClick={() => leaveChat()}
          >
            채팅방나가기
          </Button>
        </div>)
      }
    </div>
  )
}

export default ChattingAdditionalInfo