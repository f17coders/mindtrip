import { useRef, useState, useEffect } from "react";
import axios from "axios";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";
import { Card, CardBody, Button } from "@nextui-org/react";
import LeftIcon from "../../atoms/Icons/LeftIcon";
import RightIcon from "../../atoms/Icons/RightIcon";
import { villageTextColor } from "../../atoms/color";

type postitType = {
  id: string,
  postitTopicRes: {
    id: string,
    topic: string,
    postitDate: string
  },
  content: string,
  reportCount: number,
  likeCount: number,
  village: number
}


function MyPostit() {
  let accessToken = useSelector((state: RootState) => state.accessToken.value);
  let member = useSelector((state:RootState) => state.member)
  const [postitList, setPostitList] = useState<postitType[]>([]);

  const getTodayTopic = async () => {
    try {
      const response = await axios.get(
        `https://mindtrip.site/api/postits/v1/my`,
        {
          headers: {
            Authorization: accessToken,
          },
        }
      );
      console.log(response.data.result)
      setPostitList(response.data.result)
    } catch (e) {
      console.log(e);
    }
  };

  useEffect(() => {
    getTodayTopic()
  }, [])

  // 넘기기부분
  const [selectedIdx, setSelectedIdx] = useState<number>(0);

  const box = useRef<HTMLDivElement>(null);

  const getCoordinates = (event: MouseEvent | TouchEvent) => {
    if ("touches" in event && event.touches.length > 0) {
      // 터치 시작 시
      return {
        offsetX: event.touches[0].clientX,
        offsetY: event.touches[0].clientY,
      };
    } else if ("changedTouches" in event && event.changedTouches.length > 0) {
      // 터치 끝날 때
      return {
        offsetX: event.changedTouches[0].clientX,
        offsetY: event.changedTouches[0].clientY,
      };
    } else {
      // 마우스 이벤트의 경우
      return {
        offsetX: (event as MouseEvent).offsetX,
        offsetY: (event as MouseEvent).offsetY,
      };
    }
  };

  const [a1, setA1] = useState(0);
  const [a2, setA2] = useState(0);

  useEffect(() => {
    const handleMouseDown = (event: MouseEvent | TouchEvent) => {
      event.preventDefault();
      const coordinates = getCoordinates(event);
      console.log("클릭 시작 좌표:", coordinates);
      setA1(coordinates.offsetX);
    };

    const handleMouseUpOrTouchEnd = (event: MouseEvent | TouchEvent) => {
      event.preventDefault(); // 필요한 경우에만 추가
      const coordinates = getCoordinates(event);
      console.log("마우스/터치 땔 때 좌표:", coordinates);
      setA2(coordinates.offsetX);
    };

    const currentBox = box.current;
    if (currentBox) {
      currentBox.addEventListener("mousedown", handleMouseDown);
      currentBox.addEventListener("mouseup", handleMouseUpOrTouchEnd);
      currentBox.addEventListener("touchstart", handleMouseDown);
      currentBox.addEventListener("touchend", handleMouseUpOrTouchEnd);

      return () => {
        if (currentBox) {
          currentBox.removeEventListener("mousedown", handleMouseDown);
          currentBox.removeEventListener("mouseup", handleMouseUpOrTouchEnd);
          currentBox.removeEventListener("touchstart", handleMouseDown);
          currentBox.removeEventListener("touchend", handleMouseUpOrTouchEnd);
        }
      };
    }
  }, []);

  useEffect(() => {
    if (a1 - a2 > 250) {
      if (selectedIdx < postitList.length - 1) {
        setSelectedIdx((now) => now + 1);
      }
    } else if (a2 - a1 > 250) {
      if (selectedIdx > 0) {
        setSelectedIdx((now) => now - 1);
      }
    }
  }, [a1, a2]);

  return (
    <div className="flex-col p-3">
      <p className="text-center mt-3 mb-5 text-xl text-gray-600">내가 작성한 포스트잇들</p>
      {
        postitList.length != 0 && (
          <p className="text-center mb-5 h-[10vh]"><span className="text-sm" style={{fontFamily:'JamsilThin'}}>{postitList[selectedIdx].postitTopicRes.postitDate}의 질문</span><br/>{postitList[selectedIdx].postitTopicRes.topic}</p>
        )
      }
      <div>
        {
          postitList.length === 0 ? 
          (<p className="text-center text-gray-400 mt-[30vh]">아직 작성한 포스트잇이 없습니다!</p>) : 
          (<div className="relative">
            <Button 
              isIconOnly
              variant="light"
              className={`${villageTextColor[member.villageName]} absolute left-0 z-10 bottom-2`}
              onClick={()=> {
                if (selectedIdx < postitList.length - 1) {
                  setSelectedIdx((now) => now + 1);
                }
              }}
            ><LeftIcon/></Button>
            <Button 
              isIconOnly
              variant="light"
              className={`${villageTextColor[member.villageName]} absolute right-0 z-10 bottom-2`}
              onClick={()=> {
                if (selectedIdx > 0) {
                  setSelectedIdx((now) => now - 1);
                }
              }}
            ><RightIcon/></Button>
            <Card ref={box} className="h-[50vh] mt-[2vh]">
              <CardBody style={{fontFamily:'JamsilThin'}} className="p-5">
                {postitList[selectedIdx].content}
              </CardBody>
            </Card>
          </div>
          )
        }
      </div>

    </div>
  );
}

export default MyPostit;
