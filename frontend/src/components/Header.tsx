import HomeIcon from "../atoms/Icons/HomeIcon"
import MessageIcon from "../atoms/Icons/MessageIcon"
import { useNavigate } from "react-router-dom"
import {Button, Card, Badge, CardBody} from "@nextui-org/react";
import { useEffect, useState } from "react";
import EventSource from "react-native-sse";
import { useSelector } from "react-redux";
import { RootState } from "../store/store";
import { villageBackgroundColor } from "../atoms/color";
import axios from "axios";
import Swal from "sweetalert2";
import { useDispatch } from "react-redux";
import { deleteUserInfo, deleteToken } from "../store/memberSlice";

type notificationType = {
  message: string
}

function Header() {
  const navigate = useNavigate()
  const dispatch = useDispatch()
  const [openMessage, setOpenMessage] = useState<Boolean>(false)
  const [alarmCount, setAlarmCount] = useState<number>(0)
  const [notifications, setnotifications] = useState<notificationType[]|null>(null)
  const [showHomeBtn, setShowHomeBtn] = useState<boolean>(true)

  useEffect(() => {
    // 처음엔 메세지창 닫아주고
    setOpenMessage(false)
    // 서버 ON
    fetchSSE()
    

    // 만약 홈화면이면 홈버튼 없애주자
    if (window.location.pathname === '/main'){
      setShowHomeBtn(false)
    } else {
      setShowHomeBtn(true)
    }
  }, [])

  let accessToken = useSelector((state:RootState) => state.accessToken.value)
  let member = useSelector((state:RootState) => state.member)
  

  const Toast = Swal.mixin({
    toast: true,
    position:'top',
    timer: 1000
  })
  
  // 알림 서버 연결 및 데이터 가져오기
  const fetchSSE = () => {
    const eventSource = new EventSource('https://mindtrip.site/api/notifications/v1/subscribe', {
      headers: {
        Authorization: accessToken
      }
    })

    eventSource.addEventListener('open', () => {
      console.log('알림서버 열려있음')
    })

    eventSource.addEventListener('message', (e) => {
      if (e.data) {
        const parsedData = JSON.parse(e.data)
        // 처음에 count갯수가 오면 바로 보여주고
        if (parsedData.type === 'COUNT') {
          setAlarmCount(parsedData.count)
        }
        // 알림이 온다면 +1해주기
        else if (parsedData.type === 'NOTIFICATION') {
          // console.log(parsedData)
          setAlarmCount(alarmCount + 1)
          console.log(parsedData.message)
          Toast.fire({
            title: parsedData.message
          })
        }
      }
    })

    eventSource.addEventListener('error', (err) => {
      console.log(err)
      console.log('알림 서버 종료')
      eventSource.close()
      setTimeout(() => fetchSSE(), 5000)
    })
  }

  // 로딩중
  const [loading, setLoading] = useState<boolean>(false)

  // 알림 확인하면 메세지 보여주기
  const handleAlarm = function() {
    setLoading(true)
    setOpenMessage(!openMessage)
    axios.post('https://mindtrip.site/api/notifications/v1', null, {
      headers:{
        Authorization: accessToken
      }
    }).then((res) => {
      console.log(res.data)
      setLoading(false)
      setnotifications(res.data.result)
      setAlarmCount(0)
    }) .catch((err) => console.log(err))
  }


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
  
  return(
    <div className="relactive">
      <div className="h-[9vh] flex justify-between px-5 items-center">
        <Button 
          isIconOnly 
          variant="light"
          className={`${showHomeBtn ? '': 'hidden'}`}
          onClick={() => {navigate('/main')}}
        >
          <HomeIcon />
        </Button>
        <div className={`${showHomeBtn ? 'hidden': ''}`}/>
        <div>
          <Button 
            isIconOnly 
            variant="light"
            onClick={logout}
            className="ml-1"
          >
            <LogoutIcon />
          </Button>
          <Badge content={alarmCount} className={villageBackgroundColor[member.villageName]}>
            <Button 
              isIconOnly 
              variant="light"
              onClick={handleAlarm}
            >
              <MessageIcon />
            </Button>
          </Badge>
        </div>
        
        
      </div>
      {/* 메세지창 */}
      <Card 
        className={`${openMessage ? '' : 'hidden'} 
        absolute w-[80vw] h-[30vh] p-3
        bg-white top-[8vh] right-[2vw] z-20`}
      >
        <CardBody>
          {
            loading && (<p>로딩중</p>)
          }
          {
            (notifications?.length == 0) ? (<p>수신된 메세지가 없습니다!</p>) : (<div>
              {
                notifications?.map((noti, idx) => {return(
                  <div key={idx} className="py-1">{noti.message}</div>
                )})
              }
            </div>)
          }
        </CardBody>
        
      </Card>
    </div>
    
  )
}

export default Header


function LogoutIcon () {
  return(
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
  <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 9V5.25A2.25 2.25 0 0 1 10.5 3h6a2.25 2.25 0 0 1 2.25 2.25v13.5A2.25 2.25 0 0 1 16.5 21h-6a2.25 2.25 0 0 1-2.25-2.25V15m-3 0-3-3m0 0 3-3m-3 3H15" />
</svg>

  )
}