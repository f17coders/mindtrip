import MyProgress from "../../../components/Progress/Progress";
import Testcal from "../../../components/Calendar/testcal";
import { useState, useEffect } from "react";
import axios from "axios";
import { RootState } from "../../../store/store";
import { useSelector } from "react-redux";
import "react-calendar/dist/Calendar.css";

function MypageProgress() {
  const [percent, setPercent] = useState(0); // percent 상태 추가
  const [missionData, setMissionData] = useState(null); // 미션 데이터 상태 추가
  const currentDate = new Date();
  console.log(currentDate)
  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth() + 1;

  let accessToken = useSelector((state:RootState) => state.accessToken.value)
  useEffect(() => {
    // API에서 데이터 가져오는 함수
    const fetchData = async () => {
      try {
        // API 요청 시 사용할 헤더 설정
        const headers = {
          Authorization: accessToken // 회원 아이디
        };

        // API 요청 보내기
        const response = await axios.get(`https://mindtrip.site/api/missions/v1/report?year=${currentYear}&month=${currentMonth}`, { headers });
        // API 응답에서 percent 값 가져오기
        setPercent(response.data.result.percent);
        setMissionData(response.data.result.missionCalender); // 미션 데이터 설정
        console.log(missionData,'0')
      } catch (error) {
        console.error("에러: ", error); 
      }
    };

    // fetchData 함수 호출
    fetchData();
  }, [currentYear, currentMonth]); // 연도와 월이 변경될 때마다 API 요청을 다시 보내도록 설정

  return (
    <article className="bg-[#f4c2c2] flex flex-col justify-around items-center min-h-[80vh] relative">
      <div className="bg-white rounded-lg p-4 w-4/5 flex justify-around m-4">
        <MyProgress percent={percent} /> {/* percent prop으로 전달 */}
      </div>
      <span className="mb-2">월간보고서</span>
      <div className="mb-4">
        {/* TestCal 컴포넌트에 미션 데이터 전달 */}
        <Testcal missionData={missionData} />
      </div>
    </article>
  );
}

export default MypageProgress;
