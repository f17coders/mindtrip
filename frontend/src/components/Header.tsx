import HomeIcon from "../atoms/Icons/HomeIcon"
import MessageIcon from "../atoms/Icons/MessageIcon"
import { useNavigate } from "react-router-dom"
import {Button, Card, Badge} from "@nextui-org/react";
import { useEffect, useState } from "react";
import EventSource from "react-native-sse";
import { useSelector } from "react-redux";
import { RootState } from "../store/store";
import { villageBackgroundColor } from "../atoms/color";
import axios from "axios";


type notificationType = {
  message: string
}
function Header() {
  const navigate = useNavigate()
  const [openMessage, setOpenMessage] = useState<Boolean>(false)
  const [alarmCount, setAlarmCount] = useState<number>(0)
  const [notifications, setnotifications] = useState<notificationType[]|null>(null)

  useEffect(() => {
    setOpenMessage(false)
    fetchSSE()
  }, [])

  let accessToken = useSelector((state:RootState) => state.accessToken)
  let member = useSelector((state:RootState) => state.member)

  // 알림 서버 연결 및 데이터 가져오기
  const fetchSSE = () => {
    const eventSource = new EventSource('https://mindtrip.site/api/notifications/v1/subscribe', {
      headers: {
        Authorization: accessToken
      }
    })

    eventSource.addEventListener("open", () => {
      console.log('서버 연결완료')
    })

    eventSource.addEventListener('message', (e) => {
      if (e.data) {
        const parsedData = JSON.parse(e.data)
        console.log(parsedData)
        if (parsedData.type === 'COUNT') {
          setAlarmCount(parsedData.count)
        }
        else if (parsedData.type === 'NOTIFICATION') {
          setAlarmCount(alarmCount + 1)
        }
      }
    })

    eventSource.addEventListener('error', (err) => {
      console.log(err)
    })
  }


  // 알림 확인하면 메세지 보여주기
  const handleAlarm = function() {
    axios.post('https://mindtrip.site/api/notifications/v1', null, {
      headers:{
        Authorization: accessToken
      }
    }).then((res) => {
      setnotifications(res.data.result)
      setOpenMessage(!openMessage)
      setAlarmCount(0)
    }) .catch((err) => console.log(err))
  }

  
  return(
    <div className="relactive">
      <div className="h-[9vh] flex justify-between px-5 items-center">
        <Button 
          isIconOnly 
          variant="light"
          onClick={() => {navigate('/main')}}
        >
          <HomeIcon />
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
      {/* 메세지창 */}
      <Card 
        className={`${openMessage ? '' : 'hidden'} 
        absolute w-[80vw] h-[30vh] p-3
        bg-white top-[8vh] right-[2vw] z-20`}
      >
        {
          notifications?.map((noti, idx) => {return(
            <div key={idx}>{noti.message}</div>
          )})
        }
      </Card>
    </div>
    
  )
}

export default Header