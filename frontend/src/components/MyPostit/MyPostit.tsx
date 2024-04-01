import { useState, useEffect } from "react";
import moment from "moment";
import Calendar from "react-calendar";
import "./MyPostit.css";
import axios from "axios";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";

function MyPostit() {
  let accessToken = useSelector((state: RootState) => state.accessToken);

  const [selectedDate, setSelectedDate] = useState(moment()); // 선택된 날짜 상태

  const [todayPostit, setTodayPostit] = useState("");
  const [todayAnswer, setTodayAnswer] = useState("");
  const [showCalendar, setShowCalendar] = useState(false);

  // 달력 토글 함수
  const toggleCalendar = () => {
    setShowCalendar(!showCalendar);
  };

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

      setTodayPostit(`${response.data.result[0].postitTopicRes.topic}`);
      setTodayAnswer(`${response.data.result[0].content}`);
    } catch (e) {
      console.log(e);
      setTodayPostit(`${moment(date).format("YYYY-MM-DD")}에\n질문이 없습니다`);
      setTodayAnswer("");
    }
  };

  // 선택된 날짜 변경 함수
  const onChangeDate = async (date: any) => {
    if (!Array.isArray(date)) {
      setSelectedDate(moment(date));
      toggleCalendar(); // 날짜를 선택하면 달력 숨기기

      getTodayTopic(date);
    }
  };

  useEffect(() => {
    getTodayTopic(moment(Date.now()).format("YYYY-MM-DD"));
  }, []);

  return (
    <div className="my-postit-container">
      <div style={{ width: "100%", height: "100%" }}>
        {showCalendar ? (
          <>
            <div style={{ width: "70vw" }}>
              {showCalendar && (
                <Calendar onChange={(data) => onChangeDate(data)} />
              )}
            </div>
          </>
        ) : (
          <>
            <h1 className="postit-date" onClick={toggleCalendar}>
              {selectedDate.format("YYYY년 MM월 DD일")}
            </h1>
            <div>그날의 질문</div>
            <div className="relative overflow-hidden w-72 h-80 rounded-3xl cursor-pointer text-2xl font-bold bg-purple-400">
              <div className="w-full h-full flex flex-col items-center justify-center uppercase text-lg whitespace-pre-line">
                <div>
                  {todayPostit === "" ? "오늘의 질문이 없습니다" : todayPostit}
                </div>
                <div className="text-sm font-normal">
                  {todayAnswer !== "" && todayAnswer}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default MyPostit;
