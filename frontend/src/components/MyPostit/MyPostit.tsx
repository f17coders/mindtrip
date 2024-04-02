import { useRef, useState, useEffect } from "react";
import moment from "moment";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import axios from "axios";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";

function MyPostit() {
  let accessToken = useSelector((state: RootState) => state.accessToken.value);

  const [selectedDate, setSelectedDate] = useState(moment()); // 선택된 날짜 상태
  const [showCalendar, setShowCalendar] = useState(false); // 달력 표시 여부 상태

  const [todayPostit, setTodayPostit] = useState("");
  const [todayAnswer, setTodayAnswer] = useState("");
  // const [postitList, setPostitList] = useState<any[]>([]);

  const getTodayTopic = async (date: moment.MomentInput) => {
    try {
      const response = await axios.get(
        `https://mindtrip.site/api/postits/v1/my`,
        {
          headers: {
            Authorization: accessToken,
          },
        }
      );
      // setPostitList(response.data.result);
      const todayQuestion = response.data.result.find(
        (item: any) =>
          moment(item.postitTopicRes.postitDate).format("YYYY-MM-DD") ===
          moment(date).format("YYYY-MM-DD")
      );
      if (todayQuestion) {
        setTodayPostit(todayQuestion.postitTopicRes.topic);
        setTodayAnswer(todayQuestion.content);
      } else {
        setTodayPostit(
          `${moment(date).format("YYYY-MM-DD")}에\n질문이 없습니다`
        );
        setTodayAnswer("");
      }
    } catch (e) {
      console.log(e);
      setTodayPostit(`${moment(date).format("YYYY-MM-DD")}에\n질문이 없습니다`);
      setTodayAnswer("");
    }
  };

  // 넘기기부분

  const arr = [1, 2, 3, 4, 5];
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
    if (a1 - a2 > 100) {
      if (selectedIdx < arr.length - 1) {
        setSelectedIdx((now) => now + 1);
      }
    } else if (a2 - a1 > 100) {
      if (selectedIdx > 0) {
        setSelectedIdx((now) => now - 1);
      }
    }
    console.log(a1, a2, selectedIdx);
  }, [a1, a2]);

  // 선택된 날짜 변경 함수
  const onChangeDate = async (date: Date) => {
    setSelectedDate(moment(date));
    setShowCalendar(false); // 달력 숨기기
    await getTodayTopic(moment(date));
  };

  useEffect(() => {
    getTodayTopic(moment(Date.now()));
  }, []);

  return (
    <div className="my-postit-container">
      <div style={{ width: "100%", height: "100%" }}>
        <h1
          className="postit-date"
          onClick={() => setShowCalendar(!showCalendar)}
        >
          {selectedDate.format("YYYY년 MM월 DD일")}
        </h1>
        {showCalendar && (
          <DatePicker
            selected={selectedDate.toDate()}
            onChange={onChangeDate}
            dateFormat="yyyy년 MM월 dd일"
          />
        )}
        <div>{todayPostit === "" ? "그날의 질문" : todayPostit}</div>
        <div className="relative overflow-hidden w-72 h-80 rounded-3xl cursor-pointer text-2xl font-bold bg-purple-400">
          <div className="w-full h-full flex flex-col items-center justify-center uppercase text-lg whitespace-pre-line">
            {/* <div className="text-sm font-normal">
              {todayAnswer !== "" && todayAnswer}
            </div> */}
            <div
              ref={box}
              style={{
                width: "200px",
                height: "200px",
                border: "1px black",
              }}
            >
              {todayAnswer !== "" && todayAnswer}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MyPostit;
