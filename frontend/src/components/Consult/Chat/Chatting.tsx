import { useState, useEffect, useRef } from 'react';
import { Frame, Client } from '@stomp/stompjs';
import { Tooltip, Button, Input } from '@nextui-org/react';
import { changeList } from '../../../store/chatSlice';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../../../store/store';
import LeftIcon from '../../../atoms/Icons/LeftIcon'
import UpIcon from '../../../atoms/Icons/UpIcon'
import DownIcon from '../../../atoms/Icons/DownIcon'
import SendIcon from '../../../atoms/Icons/SendIcon'
import { villageBackgroundColor } from '../../../atoms/color'
import { getPersonalChat, send } from '../../../services/chat';
import ChattingAdditionalInfo from './ChattingAddtionalInfo';
import axios from 'axios';


type chatInfo = {
  title: string,
  content: string,
  closed: boolean,
  shared: boolean,
  consultId: number
}

function Chatting() {
  const dispatch = useDispatch();

  let chat = useSelector((state: RootState) => state.chat);
  const channelId = chat.selectedId;

  // 토큰
  // let memberId = useSelector((state: RootState) => state.member.memberId);
	let accessToken = useSelector((state: RootState) => state.accessToken.value);

  // 처음 열면 고민 정보 가져오기
  const [chatInfo, setChatInfo] = useState<chatInfo>({
    title: '고민제목',
    content: '고민 내용',
    closed: false,
    shared: false,
    consultId: 0
  })

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





  // 추가정보 열고 닫고
  const [show, setShow] = useState<boolean>(false)

  const [stompClient, setStompClient] = useState<Client | null>(null);
  const [newMessage, setNewMessage] = useState("");
  const [recvList, setRecvList] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [personalChat, setPersonalChat] = useState<any>({
    channelId: null,
    consultId: null,
    receiver: {
      nickname: null,
      memberId: null
    },
    sender: {
      nickname: null,
      memberId: null
    },
    messageList: []
  });

  // 스크롤바 조정
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const chatPrivateConnect = () => {

      setIsLoading(true);
  
      const stomp = new Client({

        debug: (str: string) => {
          console.log(str)
        },
        webSocketFactory: () => {
          // WebSocket을 생성하여 반환합니다.
          const socket = new WebSocket("wss://mindtrip.site/api/chat");
          return socket;
        },
        // reconnectDelay: 5000, //자동 재 연결
      });

      setIsLoading(true);

    stomp.onConnect = (frame: Frame) => { //연결이 성공하면 수행할 작업
        // setConnected(true);
      console.log('Connected: ' + frame);
      console.log(channelId)
      stomp.subscribe(`/sub/${channelId}`,
        (res: { body: string; }) => {
          setRecvList(prevRecvList => [...prevRecvList, JSON.parse(res.body)]);
        });
      console.log(recvList);
      setIsLoading(false);
    };

    stomp.activate(); //클라이언트 활성화

    //웹소켓 오류 발생 시 처리
    stomp.onWebSocketError = (error: Event) => {
      console.error('Error with websocket', error);
    };

    //연결 오류 발생 시 처리
    stomp.onStompError = (frame: Frame) => {
      console.error('Broker reported error: ' + frame.headers['message']);
      console.error('Additional details: ' + frame.body);
    };

    setStompClient(stomp);

      return () => {
        if (stomp.connected) {
            stomp.deactivate();
        }
      };
  };

  // 연결 끊기
  const disconnectChat = () => {
    if (stompClient !== null) {
      stompClient.deactivate(); // STOMP 클라이언트 비활성화
      setStompClient(null); // stompClient 상태 초기화
      setIsLoading(false); // isLoading 상태를 false로 설정
      setRecvList([]); // 받은 메시지 목록 초기화
    }
  };

  useEffect(() => {
    disconnectChat();
    if (channelId != null) {
      console.log("연결")
      chatPrivateConnect();
    }
    // 연결
    const fetchData = async () => {
      try {
        console.log("fetch")
        if (channelId != null) {
          const personalChat = await getPersonalChat(accessToken, channelId);
          setPersonalChat(personalChat);
          console.log('personalChat에 대한 정보를 출력합니다.')
          console.log(personalChat);
          setRecvList(personalChat.messageList);
        }

      } catch (error) {
        console.error("Error fetching personal chat:", error);
      }
    };
    fetchData();
  }, [channelId]);

  // 메시지 전송
  const sendMessage = () => {
    // console.log(newMessage);
    // console.log(channelId);
    if(channelId != null){
      send(stompClient, personalChat.sender, newMessage, channelId);
    }
    setNewMessage("");
  };

  useEffect(() => {
    // 채팅방이 처음 열렸을 때 스크롤을 자동으로 올립니다.
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [recvList]);

  // const formattedDate = (time: string | number | Date) =>{
  //     const createDate = new Date(time);
  //     return `${createDate.getFullYear()}-${(createDate.getMonth() + 1).toString().padStart(2, '0')}-${createDate.getDate().toString().padStart(2, '0')} ${createDate.getHours().toString().padStart(2, '0')}:${createDate.getMinutes().toString().padStart(2, '0')}`;
  // }

  //   const formattedMessage = (message : string) => {
  //       return message.split('\n').map((line, index) => (
  //           <React.Fragment key={index}>
  //               {line}
  //               <br />
  //           </React.Fragment>
  //       ));
  //   }

  const handleKeyPress = (event: any) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      // console.log(event);
      // Enter 키가 눌렸고, Shift 키가 눌리지 않았을 때
      event.preventDefault(); // 기본 동작인 폼 제출 방지
      event.stopPropagation();
      sendMessage();
    }
  };

  return (
    <div className='h-full'>
      {/* 상단 */}
      <div className='relative flex items-center justify-between mb-5'>
        <Tooltip placement='bottom' content='뒤로가기'>
          <Button
            isIconOnly
            variant='light'
            onClick={() => dispatch(changeList(true))}
          ><LeftIcon /></Button>
        </Tooltip>
        <p>{chatInfo.title}</p>
        <Button
          isIconOnly
          size='sm'
          variant='light'
          onClick={() => setShow(!show)}
        >{show ? <UpIcon /> : <DownIcon />}</Button>
      </div>

      {/* 채팅방 */}
      <div ref={scrollContainerRef} className="h-[43vh] w-full p-1 overflow-scroll">
        {
          recvList.map((msg, index) => (
            <div key={index}>
              {msg.sender.memberId === personalChat.sender.memberId ? (
                <MyBallon message={msg.text} />
              ) : (
                <OtherBallon message={msg.text} />
              )}
            </div>
          ))}   
      </div>

      {/* 채팅 치는 곳 */}
      {
        isLoading ?
          <div>채팅을 연결하는 중입니다. 잠시만 기다려주세요</div>
          :
          <div className="mt-3 flex justify-between items-center">
            <Input variant="bordered" className="w-[88%]"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyDown={handleKeyPress}
            />
            <Button isIconOnly size="sm" variant="light" onClick={sendMessage}>
              <SendIcon />
            </Button>
          </div>
      }

      {/* 추가정보 나오는 곳 */}
      {(show && chat.isMine != null) && (
        <ChattingAdditionalInfo isMine={chat.isMine} chatInfo={chatInfo}/>
      )}
    </div>
  );

}

export default Chatting;

type propsType = {
  message: string;
};

function OtherBallon({ message }: propsType) {
  return (
    <div className={`bg-gray-300 rounded-lg w-fit max-w-[90%] py-1 px-2 my-2`}>
      {message}
    </div>
  )
}

function MyBallon({ message }: propsType) {
  let member = useSelector((state: RootState) => state.member)
  return (
    <div className='flex justify-end'>
      <div className={`${villageBackgroundColor[member.villageName]} rounded-lg w-fit max-w-[90%] py-1 px-2 my-2 mr-1.5`}>
        {message}
      </div>
    </div>
  )
}
