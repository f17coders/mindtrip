import React, { useState, useEffect } from "react";
import Calendar from "react-calendar";
import "./ReactCalendar.css";
import moment from "moment";
import goodimo from "../../assets/imoticon/goodimo.png";
import sosoimo from "../../assets/imoticon/sosoimo.png";
import badimo from "../../assets/imoticon/badimo.png";

type TestCalProps = {
  missionData: MissionData | null;
}

type DayMissionData = {
  missionId:number;
  name: string;
  isFinish: boolean;
}

type DailyMissionData = {
  count: number;
  missions: DayMissionData[];
  [key: string]: any; // 인덱스 시그니처 추가
}

type MissionData = {
  [date: string] : DailyMissionData;
}

type DayData = {
  [date: string]: number;
}


const TestCal: React.FC<TestCalProps> = ({ missionData }) => {
  const [value, onChange] = useState<Date>(new Date());
  const [dayMissionData, setDayMissionData] = useState<{
    dayData: DayData;
    missionInfo: MissionData;
  }>({
    dayData: {},
    missionInfo: {},
  });
  const [selectedMissions, setSelectedMissions] = useState<string[]>([]);

  useEffect(() => {
    if (missionData) {
      console.log(missionData)
      const missionCalender = missionData;
      const tempDayData: DayData = {};
      const tempMissionInfo: MissionData = {};

      for (const date in missionCalender) {
        const count = missionCalender[date].count;
        tempDayData[date] = count;
      
        // 각 날짜의 미션 정보 설정
        const missions = missionCalender[date].missions.map((mission) => ({
          missionId: mission.missionId, // 미션 ID 추가
          name: mission.name,             // 미션 이름 추가
          isFinish: mission.isFinish
        }));
      
        tempMissionInfo[date] = {
          count: count,
          missions: missions
        };
      }

      setDayMissionData({
        dayData: tempDayData,
        missionInfo: tempMissionInfo
      });
    }
  }, [missionData]);

  const addContent = ({ date }: { date: Date }) => {
    const contents = [];
    const formattedDate = moment(date).format("YYYY-MM-DD");
    const todayDate = moment().format("YYYY-MM-DD");
  
    // 과거부터 오늘까지의 날짜에 대해서 이미지를 표시
    if (moment(formattedDate).isSameOrBefore(todayDate)) {
      const dayValue = dayMissionData.dayData[formattedDate];
  
      if (dayValue === 3) {
        contents.push(
          <img
            key={formattedDate}
            src={goodimo}
            className="diaryImg"
            width="26"
            height="26"
            alt="good day"
          />
        );
      } else if (dayValue === 2 || dayValue === 1) {
        contents.push(
          <img
            key={formattedDate}
            src={sosoimo}
            className="diaryImg"
            width="26"
            height="26"
            alt="so-so day"
          />
        );
      } else {
        contents.push(
          <img
            key={formattedDate}
            src={badimo}
            className="diaryImg"
            width="26"
            height="26"
            alt="bad day"
          />
        );
      }
    }
    return contents;
  };

  const handleDayClick = (date: Date) => {
    const formattedDate = moment(date).format("YYYY-MM-DD");
    const missions = dayMissionData.missionInfo[formattedDate]?.missions || [];

    if (missions.length === 0) {
      setSelectedMissions(["해당 날짜에 미션이 없습니다."]);
    } else {
      const missionStatus = missions.map((mission) => {
        return `${mission.name}: ${mission.isFinish ? "성공" : "실패"}`;
      });

      setSelectedMissions(missionStatus);
    }
  };

  return (
    <>
      <Calendar
        onChange={(date) => onChange(date as Date)}
        value={value}
        next2Label={null}
        prev2Label={null}
        // @ts-ignore
        formatDay={(locale, date) => moment(date).format("D")} 
        // @ts-ignore
        formatShortWeekday={(locale, date) => moment(date).format("dd")} 
        tileContent={addContent}
        showNeighboringMonth={false}
        onClickDay={handleDayClick}
      />
      {selectedMissions.length > 0 && (
        <div className="mt-4 text-center bg-white w-4/5 p-4 rounded-lg border border-gray-300 mx-auto">
          <p>선택한 날짜의 미션 정보:</p>
          {selectedMissions.map((mission, index) => (
            <p key={index}>{mission}</p>
          ))}
        </div>
      )}
    </>
  );
}

export default TestCal