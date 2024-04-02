import { useState, useEffect } from "react";
import axios from "axios";
import { Link } from 'react-router-dom';
import Successbtn from '../../atoms/buttons/successbtn';
import MissionTree from "../../components/Loading/MissionTree";
import Header from "../../components/Header";
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store/store';
import { countUpdate } from "../../store/memberSlice";
import Swal from "sweetalert2";
import profile1 from "../../assets/levelupbadge/씨앗.png";
import profile2 from "../../assets/levelupbadge/새싹.png";
import profile3 from "../../assets/levelupbadge/잎새.png";
import profile4 from "../../assets/levelupbadge/나뭇가지.png";
import profile5 from "../../assets/levelupbadge/나무.png";
import profile6 from "../../assets/levelupbadge/열매.png";


// 미션 타입 정의
type Mission = {
  missionId: number;
  name: string;
  isFinish: boolean;
};

function Mission() {
  const dispatch = useDispatch()
  // 미션 데이터 상태
  const [missions, setMissions] = useState<Mission[]>([]); // 미션 타입을 명시

  const [isMidnight, setIsMidnight] = useState<boolean>(false);
  let accessToken = useSelector((state: RootState) => state.accessToken.value);
  let member = useSelector((state: RootState) => state.member);

  const [ProfileImg, setProfileImg] = useState(profile1);

  const changeProfile = () => {
    if (member.level === 1) {
      setProfileImg(profile2);
    }
    if (member.level === 2) {
      setProfileImg(profile3);
    }
    if (member.level === 3) {
      setProfileImg(profile4);
    }
    if (member.level === 4) {
      setProfileImg(profile5);
    }
    if (member.level === 5) {
      setProfileImg(profile6);
    }
    if (member.level === 6){
      setProfileImg(profile6)}
  };

  // 페이지가 로드될 때 미션 데이터를 가져오는 useEffect 훅
  useEffect(() => {
    const fetchMissions = async () => {
      try {
        // 미션 데이터를 가져오는 GET 요청
        const response = await axios.get(
          "https://mindtrip.site/api/missions/v1/mytable",
          {
            headers: {
              Authorization: accessToken,
            },
          }
        );
        console.log("미션", response.data);
        // 가져온 미션 데이터를 상태에 설정
        setMissions(response.data.result);
      } catch (error) {
        console.error("Error fetching missions:", error);
      }
    };

    // 함수 실행
    fetchMissions();

    // 시간확인
    const currentTime = new Date();
    const currentHour = currentTime.getHours();
    const currentMinute = currentTime.getMinutes();
    console.log(currentHour);
    console.log(currentMinute);

    if (currentHour === 0 && currentMinute >= 0 && currentMinute <= 5) {
      setIsMidnight(true);
    }
  }, []);
  useEffect(() => {
    changeProfile();
  }, [member.level]);

  // 미션 성공 처리 함수
  const handleMissionSuccess = async (missionId: number) => {
    try {
      // 해당 미션의 isFinish를 true로 변경하는 POST 요청
      const response = await axios.post(
        `https://mindtrip.site/api/missions/v1/mytable/${missionId}`,
        {},
        {
          headers: {
            Authorization: accessToken,
          },
        }
      );

      const successCount = await axios.get(
        "https://mindtrip.site/api/members/v1/mission-count",
        {
          headers: {
            Authorization: accessToken,
          },
        }
      );
      console.log("Success:", response.data);

      // 상태 업데이트: 해당 미션의 isFinish를 true로 변경
      setMissions((prevMissions) =>
        prevMissions.map((mission) =>
          mission.missionId === missionId
            ? { ...mission, isFinish: true }
            : mission
        )
      );
      console.log("성공개수:",successCount.data.result)
      dispatch(countUpdate())

      if (successCount.data.result === 3) {
        // 미션 성공 메세지 띄워주기  \
        Swal.fire({
          title: "레벨업!",
          text: "새싹으로 레벨업 했습니다.",
          imageUrl: ProfileImg,
          imageWidth: 200,
          imageHeight: 200,
          imageAlt: "새싹"
        });
      }
      if (successCount.data.result === 9) {
        // 미션 성공 메세지 띄워주기  \
        Swal.fire({
          title: "레벨업!",
          text: "잎새로 레벨업 했습니다.",
          imageUrl: ProfileImg,
          imageWidth: 200,
          imageHeight: 200,
          imageAlt: "잎새"
        });
      }
      if (successCount.data.result === 15) {
        // 미션 성공 메세지 띄워주기  \
        Swal.fire({
          title: "레벨업!",
          text: "나뭇가지로 레벨업 했습니다.",
          imageUrl: ProfileImg,
          imageWidth: 200,
          imageHeight: 200,
          imageAlt: "나뭇가지"
        });
      }
      if (successCount.data.result === 21) {
        // 미션 성공 메세지 띄워주기  \
        Swal.fire({
          title: "레벨업!",
          text: "나무로 레벨업 했습니다.",
          imageUrl: ProfileImg,
          imageWidth: 200,
          imageHeight: 200,
          imageAlt: "나무"
        });
      }
      if (successCount.data.result === 30) {
        // 미션 성공 메세지 띄워주기  \
        Swal.fire({
          title: "레벨업!",
          text: "열매로 레벨업 했습니다. 레벨업 최대치에 도착했습니다!",
          imageUrl: ProfileImg,
          imageWidth: 200,
          imageHeight: 200,
          imageAlt: "열매"
        });
      }
    } catch (error) {
      console.error("Error updating mission:", error);
    }
  };

  return (
    <div className="bg-[#fff7e0] px-2 py-8 h-screen ">
      <Header />
      <div className="flex flex-col justify-center items-center mb-6">
        <h1 className="text-5xl font-bold mt-20">오늘의 미션</h1>
      </div>
      {isMidnight ? (
        <div>
          <MissionTree />
          <span className="block text-center">미션을 불러오는 중입니다...</span>
        </div>
      ) : (
        <div>
          <div className="flex justify-end text-sm mb-4 w-4/5 mx-auto">
            <span>
              달성률: {missions.filter((mission) => mission.isFinish).length}/
              {missions.length}
            </span>
          </div>
          <div className="space-y-4">
            {/* 미션 데이터를 반복하여 출력 */}
            {missions.map((mission) => (
              <div
                key={mission.missionId}
                className="flex justify-between items-center p-2 bg-white rounded-lg shadow w-4/5 mx-auto"
              >
                {/* 미션 이름 출력 */}
                <span className="text-lg">{mission.name}</span>
                {/* 성공 버튼 */}
                <Successbtn
                  isFinish={mission.isFinish}
                  missionId={mission.missionId}
                  onClick={handleMissionSuccess}
                />
              </div>
            ))}
          </div>
          <div className="mt-8 p-2 rounded-lg text-center w-4/5 mx-auto">
            <p className="mb-4">오늘의 미션이 맘에 들지 않았다면,</p>
            {/* 수정하러 가는 링크 */}
            <Link
              to="/fixmission"
              className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-700 mt-4"
            >
              수정하러가기
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

export default Mission;
