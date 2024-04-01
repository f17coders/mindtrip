import  { useState, useEffect } from "react";
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
        (item: any) => moment(item.postitTopicRes.postitDate).format("YYYY-MM-DD") === moment(date).format("YYYY-MM-DD")
      );
      if (todayQuestion) {
        setTodayPostit(todayQuestion.postitTopicRes.topic);
        setTodayAnswer(todayQuestion.content);
      } else {
        setTodayPostit(`${moment(date).format("YYYY-MM-DD")}에\n질문이 없습니다`);
        setTodayAnswer("");
      }
    } catch (e) {
      console.log(e);
      setTodayPostit(`${moment(date).format("YYYY-MM-DD")}에\n질문이 없습니다`);
      setTodayAnswer("");
    }
  };

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
        <h1 className="postit-date" onClick={() => setShowCalendar(!showCalendar)}>
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
            <div className="text-sm font-normal">
              {todayAnswer !== "" && todayAnswer}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MyPostit;
