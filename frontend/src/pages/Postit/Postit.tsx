import React, { useState, useEffect } from "react";
import PostIt from "../../atoms/postit/postititem";
import PostitModal from "../../components/MyPostit/PostitModal";
import axios, { AxiosError } from "axios";
import Header from "../../components/Header";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";
import Swal from "sweetalert2";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import styles from "../../components/Calendar/Calendar.module.css";

const PostitPage: React.FC = () => {
  const colors = ["#ffff88", "#ffcc00", "#ff9999", "#99ccff"];
  const [postits, setPostits] = useState<any[]>([]);
  const [topic, setTopic] = useState<string>("");
  const [topicId, setTopicId] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  
  //  api에서 오늘 날짜 가져오는 부분
  const currentDate = new Date();
  const year = currentDate.getFullYear();
  const month = (currentDate.getMonth() + 1).toString().padStart(2, "0"); // 1월: "01", 11월: "11"
  const day = currentDate.getDate().toString().padStart(2, "0"); // 1일: "01", 11일: "11"
  const formattedDate = `${year}-${month}-${day}`;
  console.log(formattedDate);
  
  //datepicker 관련
  const [startDate, setStartDate] = useState<Date | null>(currentDate);
  console.log(startDate,"시작날짜")
  
  let member = useSelector((state: RootState) => state.member);
  let accessToken = useSelector((state: RootState) => state.accessToken);

  const fetchData = async () => {
    try {
      console.log("gi");
      const response = await axios.get(
        `https://mindtrip.site/api/postits/v1?date=${formattedDate}&order=like&village=all&page=0&size=10`,
        {
          headers: {
            Authorization: accessToken,
          },
        }
      );
      console.log("feat");
      console.log(response.data);
      setPostits(response.data.result.postitResList);
      setTopic(response.data.result.topic);
      setTopicId(response.data.result.topicId);
    } catch (error) {
      console.log("Error fetching missions:", error);
    }
  };

  useEffect(() => {
    console.log("useEffect");
    fetchData();
  }, []);

  const addPostit = async (content: string) => {
    try {
      console.log("post");
      const response = await axios.post(
        "https://mindtrip.site/api/postits/v1",
        {
          topicId: topicId,
          postitDate: formattedDate,
          content: content,
          village: member.villageName,
        },
        {
          headers: {
            Authorization: accessToken,
          },
        }
      );
      console.log(response.data);
      setIsModalOpen(false); // 새로운 포스트잇을 추가한 후에 모달을 닫기
      fetchData(); // 모달을 닫은 후에 포스트잇 목록을 다시 불러오기
    } catch (error: any) {
      console.log(error.response?.data);

      let errorMessage;
      switch (error.response?.data?.code) {
        case "G011":
          errorMessage = "내용은 필수입니다.";
          break;
        case "B300":
          errorMessage = "존재하지 않는 주제입니다.";
          break;
        case "B302":
          errorMessage = "이미 포스트잇을 등록하였습니다.";
          break;
        case "B303":
          errorMessage = "오늘의 주제만 등록과 삭제가 가능합니다.";
          break;
        default:
          errorMessage = "알 수 없는 오류가 발생했습니다.";
      }

      Swal.fire({
        text: errorMessage,
        icon: "warning",
      });

      console.log("새 포스트잇 에러 :", error);
    }
  };

  const handleFirstPostitClick = () => {
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  const handleNewPostSubmit = (content: string) => {
    addPostit(content);
  };

  return (
    <div className="bg-[#fff7e0] px-2 min-h-screen">
      <Header />
      <div className="flex flex-col justify-center items-center mb-6">
        <div className="flex flex-row">
          <DatePicker
            dateFormat="yyyy.MM.dd"
            className={styles.datePicker}
            selected={startDate}
            onChange={(date) => date && setStartDate(date)}
          />
          <h1 className="text-center flex items-center justify-center">의 질문 보기</h1>
        </div>
        <h1 className="text-xl font-bold mt-10 w-4/5">{topic}</h1>
      </div>
      <div className="bg-white rounded-lg shadow-md p-6 mb-8 w-4/5 mx-auto">
        <div className="flex justify-center items-center flex-wrap list-none">
          <div className="m-2">
            <PostIt
              color={colors[2]}
              onClick={handleFirstPostitClick}
              style={{ transition: "transform 0.3s ease-in-out" }}
            >
              눌러서 대답하기
            </PostIt>
          </div>
          {postits.map((postit) => (
            <div className="m-2" key={postit.id}>
              <PostIt color={postit.color}>{postit.content}</PostIt>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-8 p-2 rounded-lg text-center w-4/5 mx-auto"></div>
      <PostitModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        onSubmit={handleNewPostSubmit}
      />
    </div>
  );
};

export default PostitPage;
