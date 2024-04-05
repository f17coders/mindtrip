import { useState, useEffect } from "react";
import axios from "axios";
import FixMissionBox from "../../components/missonbox/FixMissionBox";
import { useNavigate } from "react-router-dom";
import { Button } from "@nextui-org/react";
import { useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import { villageBackgroundColor} from '../../atoms/color';


type Mission = {
  missionId: number;
  name: string;
  isFinish: boolean;
}

type CategoryMission = {
  category: string;
  missionBaseResList: Mission[];
}

type AllMissionsResponse = {
  result: {
    categoryMissionResList: CategoryMission[];
  };
}

type TodayMissionsResponse =  {
  result: Mission[];
}

function Fixmission() {
  const [missions, setMissions] = useState<CategoryMission[]>([]);
  const [todayMissions, setTodayMissions] = useState<Mission[]>([]);
  const navigate = useNavigate();
  let accessToken = useSelector((state:RootState) => state.accessToken.value)
  let member = useSelector((state:RootState) => state.member)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const allMissionsResponse = await axios.get<AllMissionsResponse>(
          "https://mindtrip.site/api/missions/v0"
        );
        setMissions(allMissionsResponse.data.result.categoryMissionResList);

        const todayMissionsResponse = await axios.get<TodayMissionsResponse>(
          "https://mindtrip.site/api/missions/v1/mytable",
          {
            headers: {
              Authorization:accessToken               // 사용자 ID를 적절히 설정하세요.
            }
          }
        );
        setTodayMissions(todayMissionsResponse.data.result);
      } catch (error) {
        console.error("Error fetching missions:", error);
      }
    };

    fetchData();
  }, []);

  const addMissionToToday = async (mission: Mission): Promise<void> => { // Promise<void>로 반환 타입 수정
    // 중복된 미션인지 확인
    if (todayMissions.find((item) => item.missionId === mission.missionId)) {
      console.log("이미 추가된 미션입니다.");
      return;
    }
  
    // 최대 3개까지만 추가 가능
    if (todayMissions.length >= 3) {
      alert("최대 3개까지만 추가할 수 있습니다.");
      return;
    }
  
    // 새로운 미션 객체 생성
    const newMission: Mission = {
      missionId: mission.missionId,
      name: mission.name,
      isFinish: false, // 항상 false로 설정
    };
  
    // 새로운 미션 추가
    setTodayMissions((prevMissions) => [...prevMissions, newMission]);
  };
  

  const removeMissionFromToday = (mission: Mission) => {
    if (mission.isFinish) {
      console.log("완료된 미션은 삭제할 수 없습니다.");
      return;
    }

    setTodayMissions((prevMissions) =>
      prevMissions.filter((item) => item.missionId !== mission.missionId)
    );
  };

  const handleFixCompleted = async () => {
    try {
      const requestData = Object.fromEntries(
        todayMissions.map((mission) => [mission.missionId.toString(), mission])
      );

      await axios.put(
        "https://mindtrip.site/api/missions/v1/mytable",
        requestData,
        {
          headers: {
            Authorization: accessToken// 사용자 ID를 적절히 설정하세요.
          }
        }
      );
      navigate("/mission")
      console.log("오늘의 미션 수정이 완료되었습니다.");
    } catch (error) {
      console.log("Error updating today's missions:", error);
    }
  };

  return (
    <div className={`${villageBackgroundColor[member.villageName]} px-2 py-8 min-h-screen`}>
      <div className="grid gap-8 flex items-center">
        <div className=" flex justify-center">
        <div
          className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-700 mt-4"
          onClick={handleFixCompleted} // 수정 완료 버튼 클릭 시 호출될 함수 추가
        >
          수정 완료
        </div>
      </div>
        
        <div className="bg-[#f4f4f4] p-4 rounded-lg mb-4 border-2 border-gray-500 shadow-lg min-h-560">
          <h2 className="text-xl font-semibold mb-4">오늘의 미션</h2>
          <div className="space-y-4">
            {todayMissions.map((mission) => (
              <div
                key={mission.missionId}
                className="flex justify-between items-center"
              >
                <p>{mission.name}</p>
                <Button
                  radius="full"
                  className="bg-gradient-to-tr from-pink-500 to-yellow-500 text-white shadow-lg"
                  onClick={() => removeMissionFromToday(mission)}
                  disabled={mission.isFinish}
                >
                  {mission.isFinish ? "완료" : "삭제"}
                </Button>
              </div>
            ))}
          </div>
        </div>
        <div
          className="allmission shadow-lg rounded-lg overflow-y-auto"
          style={{ maxHeight: "70vh" }}
        >
          {missions.map((categoryMission, index) => (
            <FixMissionBox
              key={index}
              category={categoryMission.category}
              missions={categoryMission.missionBaseResList}
              addMissionToToday={addMissionToToday}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default Fixmission;
