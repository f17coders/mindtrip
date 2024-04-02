import { useEffect, useState } from 'react'
import { changeList, changeSelectedId, setisMine } from '../../../store/chatSlice'
import { useDispatch, useSelector } from "react-redux"
import { RootState } from '../../../store/store'
import { villageBackgroundColor } from '../../../atoms/color'
import { chattingRoom } from '../../../types/DataTypes'
import { loadChattingMine, loadChattingOthers } from '../../../api/consults'

// 채팅방 리스트들이 들어가는 컴포넌트

function ChatList() {
  // 상단 탭 제어용 true면 나의 고민
  const [pickFirst, setPickFirst] = useState<boolean>(true)
  // const [personalChatId, setPersonalChatId] = useState<string>("")

  let member = useSelector((state: RootState) => state.member)
  let accessToken = useSelector((state: RootState) => state.accessToken.value)

  const [myChattings, setMyChattings] = useState<chattingRoom[]>([])
  // 내 채팅 불러오는 함수
  const loadMyChatting = async () => {
    try {
      let tempList: chattingRoom[] | null = await loadChattingMine(accessToken)
      if (tempList) {
        setMyChattings(tempList)
      }
    } catch (err) {
      console.log(err)
    }
  }

  const [otherChattings, setOtherChattings] = useState<chattingRoom[]>([])
  // 내 채팅 불러오는 함수
  const loadOtherChatting = async () => {
    try {
      let tempList: chattingRoom[] | null = await loadChattingOthers(accessToken)
      if (tempList) {
        setOtherChattings(tempList)
      }
    } catch (err) {
      console.log(err)
    }
  }

  useEffect(() => {
    loadMyChatting()
    loadOtherChatting()
  }, [pickFirst])


  return (
    <div className="h-[70vh]">
      <div className="flex h-[8vh] justify-between items-center text-center">
        <div className="w-1/2 hover:cursor-pointer" onClick={() => setPickFirst(true)}>나의 고민</div>
        <div className="w-1/2 hover:cursor-pointer" onClick={() => setPickFirst(false)}>내가 들어준 고민</div>
      </div>
      <div
        className={`${villageBackgroundColor[member.villageName]} border-2 border-gray-200 w-1/2 h-1 rounded-md mb-2`}
        style={{ transition: 'margin-left 0.3s ease', marginLeft: pickFirst ? 0 : '50%' }}
      />
      {
        pickFirst && (
          <div className='overflow-y-auto h-full pb-[200px]'>
            {
              (myChattings === null || myChattings.length === 0) ? (<p>고민을 아직 생성하지 않았거나, 참여한 사람이 없습니다!</p>) : (<>{
                myChattings.map((chatting, idx) => {
                  return (
                    <div key={idx} >
                      {
                        chatting.shared === false && (<Chatting title={chatting.title} content={chatting.text} channelId={chatting.channelId} isMine={true} />)
                      }
                    </div>
                  )
                })
              }</>)
            }
          </div>
        )
      }
      {
        pickFirst === false && (
          <div className='overflow-y-auto h-full pb-[200px]'>
            {
              otherChattings?.map((chatting, idx) => {
                return (
                  <div key={idx}>
                    {
                      chatting.shared === false && (<Chatting title={chatting.title} content={chatting.text} channelId={chatting.channelId} isMine={false}/>)
                    }
                  </div>

                )
              })
            }
            {
              (otherChattings === null || otherChattings.length == 0) && (<p>아직 참여한 채팅이 없습니다!</p>)
            }
          </div>
        )
      }
    </div>
  )
}

export default ChatList

type propstype = {
  readonly title: string,
  readonly content: string | null,
  readonly channelId: string // channelId 추가,
  readonly isMine: boolean
}

function Chatting({ title, content, channelId, isMine }: propstype) {
  const dispatch = useDispatch()

  const handleClick = (channelId: string) => {
    dispatch(changeList(false))
    dispatch(changeSelectedId(channelId))
    dispatch(setisMine(isMine))
  };

  return (
    <div className="relative border-b h-20 p-3 hover:bg-gray-100" onClick={() => handleClick(channelId)}>
      <p className="text-lg">{title}</p>
      <p className="text-sm overflow-hidden">{content}</p>
    </div>
  )
}
