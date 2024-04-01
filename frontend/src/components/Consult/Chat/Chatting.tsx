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
// 채팅방
import { getPersonalChat, send } from '../../../services/chat';
import ChattingAdditionalInfo from './ChattingAddtionalInfo';
import SockJS from "sockjs-client";


function Chatting() {
  const dispatch = useDispatch();

  let chat = useSelector((state: RootState) => state.chat);
  const channelId = chat.selectedId;
  console.log(channelId);

  // 토큰
  // let memberId = useSelector((state: RootState) => state.member.memberId);
	let accessToken = useSelector((state: RootState) => state.accessToken.value);

  // 추가정보 열고 닫고
  const [show, setShow] = useState<boolean>(false)

  const [stompClient, setStompClient] = useState<Client | null>(null);
  // const [connected, setConnected] = useState<boolean>(false);
  // const [name, setName] = useState<string>('');
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


// 이 방이 내가 만든방인지 / 참여한 방인지를 확인하기 위한 변수
  let member = useSelector((state: RootState) => state.member)
  const [isMine, setIsMine] = useState<Boolean|null>(true)

  useEffect(() => {
    // 만약 채팅방의 user Id와 내 아이디가 같다면 내꺼임
    if (member.memberId == personalChat.sender.memberId) {
      console.log('memberid')
      console.log(personalChat.sender.memberId)
      console.log(member.memberId)
      setIsMine(true)
    } else {
      console.log('memberid')
      console.log(personalChat.sender.memberId)
      console.log(member.memberId)
      setIsMine(false)
    }
  }, [personalChat])


  // 스크롤바 조정
  const scrollContainerRef = useRef<HTMLDivElement>(null);

    const chatPrivateConnect = () => {
      const serverURL = `https://mindtrip.site/api/chat`;
      let socket = new SockJS(serverURL);
      const stomp = new Client({
        // brokerURL: 'ws://localhost:8000/api/chat',
        // brokerURL: 'wss://mindtrip.site/api/chat',
        
        // connectHeaders: {
        //   Authorization: `${accessToken}`,
        // },

        debug: (str: string) => {
          console.log(str)
        },
        webSocketFactory: () => socket
        // reconnectDelay: 5000, //자동 재 연결
      });

      
      
      setIsLoading(true);

    stomp.onConnect = (frame: Frame) => { //연결이 성공하면 수행할 작업
      //   setConnected(true);
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

    //   return () => {
    //     if (stomp.connected) {
    //         stomp.deactivate();
    //     }
    //   };
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
    //   readPersonalChat(accessToken, personalChat.personalChatId);
    setNewMessage("");
  };

  useEffect(() => {
    // 채팅방이 처음 열렸을 때 스크롤을 자동으로 올립니다.
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [recvList]);

  // const formattedDate = (time) =>{
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

  // const connect = () => {
  //   setConnected(true)
  //     if (stompClient) {
  //         stompClient.activate();
  //     }else{
  //       console.error('연결 실패: Stomp 클라이언트가 연결되지 않았습니다.');
  //     }
  // };

  // const disconnect = () => {
  //     if (stompClient) {
  //         stompClient.deactivate();
  //         setConnected(false);
  //         console.log('Disconnected');
  //     }
  // };

  // const sendName = (message : string) => {
  //     if (stompClient) {
  //       console.log(message)
  //         stompClient.publish({
  //             destination: '/pub/api/chat/send/{channelId}',
  //             // body: JSON.stringify({ name: name })
  //         });
  //     }
  // };


  // const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
  //     setName(event.target.value);
  // };

  // const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
  //     event.preventDefault();
  //     sendName(event.target.value);
  // };

  //     const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
  //       setMessage(event.target.value); // 입력 필드의 값을 상태에 업데이트합니다.
  //       console.log(event.target.value);
  //   };

  // 고민 종료하는 함수

  return (
    <div>
      {/* 상단 */}
      <div className='relative flex items-center justify-between mb-5'>
        <Tooltip placement='bottom' content='뒤로가기'>
          <Button
            isIconOnly
            variant='light'
            onClick={() => dispatch(changeList(true))}
          ><LeftIcon /></Button>
        </Tooltip>
        <p>고민 제목</p>
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
      {(show && isMine != null) && (
        <ChattingAdditionalInfo isMine={isMine}/>
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
