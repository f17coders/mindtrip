// Postititem.tsx
import React, { useState } from "react";
import "./postit.css";
import PostitLikeBtn from "../buttons/PostitLikeBtn";
// import { Button } from '@nextui-org/react';
import { postitType } from "../../pages/Postit/Postit";
import axios from "axios";
import Swal from "sweetalert2";
// import ReportIcon from '../Icons/ReportIcon';
import ReportBtn from "../buttons/ReportBtn";

type PostitProps = {
  accessToken: string;
  postItem?: postitType;
  color: string;
  onClick?: () => void; // onClick prop을 선택적으로(Optional) 처리
  children: string;
  style?: React.CSSProperties;
  firstData?: boolean;
};

const Postit: React.FC<PostitProps> = ({
  accessToken,
  postItem,
  color,
  onClick,
  children,
  style,
  firstData,
}) => {
  const [isLike, setIsLike] = useState((postItem && postItem.isLike) || false);
  const [isReport, setIsReport] = useState(
    (postItem && postItem.isReport) || false
  );
  const postitStyle: React.CSSProperties = {
    ...style, // 전달받은 style을 포함합니다.
  };

  const LikeHandle = async (clickData: boolean) => {
    try {
      if (clickData) {
        console.log(clickData)
        await axios.post(
          `https://mindtrip.site/api/postits/v1/like/${postItem?.id}`,
          {},
          {
            headers: {
              Authorization: accessToken,
            },
          }
        );
      } else {
        await axios.delete(
          `https://mindtrip.site/api/postits/v1/like/${postItem?.id}`,
          {
            headers: {
              Authorization: accessToken,
            },
          }
        );
      }
    } catch (error) {
      console.log(error);
    }
    setIsLike(clickData);
  };

  const ReportHandle = async () => {
    try {
      if (!isReport) {
        Swal.fire({
          title: "신고 하시겠습니까?",
          showDenyButton: true,
          confirmButtonText: "확인",
          denyButtonText: `취소`,
        }).then(async (result) => {
          if (result.isConfirmed) {
            await axios.post(
              `https://mindtrip.site/api/postits/v1/report/${postItem?.id}`,
              {},
              {
                headers: {
                  Authorization: accessToken,
                },
              }
            );
            Swal.fire({ text: "신고가 정상적으로 완료되었습니다" });
            setIsReport(true);
          } else if (result.isDenied) {
            Swal.fire({ text: "취소되었습니다", icon: "info" });
          }
        });
      } else {
        Swal.fire({
          title: "신고를 취소 하시겠습니까?",
          showDenyButton: true,
          confirmButtonText: "확인",
          denyButtonText: `취소`,
        }).then(async (result) => {
          if (result.isConfirmed) {
            await axios.delete(
              `https://mindtrip.site/api/postits/v1/report/${postItem?.id}`,
              {
                headers: {
                  Authorization: accessToken,
                },
              }
            );
            Swal.fire({ text: "신고 취소가 정상적으로 완료되었습니다" });
            setIsReport(false);
          } else if (result.isDenied) {
            Swal.fire({ text: "취소되었습니다", icon: "info" });
          }
        });
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className={`${color} rgyPostIt`} style={postitStyle} onClick={onClick}>
      {!firstData && (
        <div>
          <div style={{ position: "absolute", bottom: 10, right: 50 }}>
            <PostitLikeBtn isLike={isLike} setIsLike={LikeHandle} />
          </div>
          <div style={{ position: "absolute", bottom: 5, right: 10 }}>
            {postItem && postItem?.reportCount < 5 && (
              <ReportBtn  setReport={ReportHandle} isReport={isReport}/>
            )}
          </div>
        </div>
      )}
      <p>{children}</p>
    </div>
  );
};

export default Postit;
