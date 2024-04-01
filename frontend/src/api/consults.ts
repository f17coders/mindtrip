import axios from "axios";
import { consultType, categoryType, chattingRoom } from "../types/DataTypes";
import Swal from "sweetalert2";

// 입장가능한 고민 목록 가져오기
async function getConsults(token: string): Promise<consultType[]> {
  try {
    const res = await axios.get('https://mindtrip.site/api/consults/v1/available', {
      headers: {
        Authorization: token
      }
    });
    return res.data.result.consultList;
  } catch (err) {
    console.log(err);
    return [];
  }
}


// 고민 등록하기
async function uploadConsult(token:string, title: string, content: string, categoryId: number | string | categoryType): Promise<void> {
  try {
    await axios.post('https://mindtrip.site/api/consults/v1', {
      'title': title,
      'content': content,
      'categoryId': categoryId
    }, {
      headers: {
        Authorization: token
      }
    })
    Swal.fire({
      title: '등록완료~',
      icon: 'success'
    }).then(() => {
      location.reload()
    })
  } catch (err) {
    console.log(err)
  }
}

// 카테고리 가져오기
async function getCategory(token:string): Promise<categoryType[]> {
  try {
    const res = await axios.get('https://mindtrip.site/api/consults/v1/category', {
      headers: {
        Authorization: token
      }
    })
    return res.data.result.consultCategoryList
  } catch (err) {
    console.log(err)
    return []
  }
}


// 공유된 고민 가져오기
// 입장가능한 고민 목록 가져오기
async function getSharedConsult(token: string): Promise<consultType[]> {
  try {
    const res = await axios.get('https://mindtrip.site/api/consults/v1/shared/1', {
      headers: {
        Authorization: token
      }
    });
    return res.data.result.consultList;
  } catch (err) {
    console.log(err);
    return [];
  }
}

// 채팅방에 입장하기
async function enterRoom(token: string, consultId: number): Promise<string|null> {
  try {
    const res = await axios.post(`https://mindtrip.site/api/channels/v1/enter/${consultId}`, null, {
      headers: {
        Authorization: token
      }
    })
    return res.data.result
  } catch (err :any) {
    switch (err.response?.data?.code) {
      case 'B100':
        Swal.fire({
          text: '권한이 없는 사용자입니다.'
        })
        break
      default:
      console.log(err)
    }
    return null
  }
}


// 대화중인 채팅 목록 불러오기(내꺼)
async function loadChattingMine(token:string) :Promise<chattingRoom[]|null>{
  try{
    const res = await axios.get('https://mindtrip.site/api/consults/v1/mine',{
      headers: {
        Authorization: token
      }
    })
    const tempRes = res.data.result
    if (tempRes) {
      return tempRes.consultChattingRes
    } else {
      return null
    }
  } catch (err) {
    console.log(err)
    return null
  }
}


// 대화중인 채팅 목록 불러오기(다른사람꺼)
async function loadChattingOthers(token:string) :Promise<chattingRoom[]|null>{
  console.log(token)
  try{
    const res = await axios.get('https://mindtrip.site/api/consults/v1/others',{
      headers: {
        Authorization: token
      }
    })
    console.log(res)
    const temp = res.data.result
    if (temp === null) {
      return null
    } else {
      return temp
    }
  } catch (err) {
    console.log(err)
    return null
  }
}

export { getConsults, uploadConsult, getCategory, getSharedConsult, enterRoom, loadChattingMine, loadChattingOthers }