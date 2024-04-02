import axios from 'axios'
import { villageNameType, pictureResultType, allSurveys } from '../types/DataTypes'


// 설문조사 항목 가져오기
async function getSurvey() :Promise<allSurveys|null> {
  try {
    const res  = await axios.get('https://mindtrip.site/api/htp/v0/question/all')
    // console.log(res.data)
    return res.data
  } catch (err) {
    console.log(err)
    return null
  }
}


// 검사 결과 문장으로 가져오기
async function getResult1(token: string): Promise<string|null> {
  try {
    const res = await axios.get('https://mindtrip.site/api/htp/v1/result/sentence', {
      headers: {
        Authorization: token
      }
    })
    return res.data.data
  } catch (err) {
    console.log(err)
    return null
  }
}


// 검사 결과 마을 가져오기
async function getResult2(token: string): Promise<villageNameType|null> {
  try {
    const res = await axios.get('https://mindtrip.site/api/htp/v1/result/village', {
      headers: {
        Authorization: token
      }
    });
    return res.data.data
  } catch (err) {
    console.log(err)
    return null
  }
}


// 마을 한글로 바꿔주기
function changeLang(village:string) :string {
  if (village === 'apple') {
    return '사과'
  } else if (village === 'orange') {
    return '오렌지'
  } else if (village === 'pineapple') {
    return '파인애플'
  } else if (village === 'watermelon') {
    return '수박'
  } else if (village === 'grape') {
    return '포도'
  } else if (village === 'peach') {
    return '복숭아'
  }else if (village === 'blueberry') {
    return '블루베리'
  } else {
    return '카카오'
  }
}


// 마이페이지에서 결과 조회하기(문장)
async function loadRecentResult(token:string) :Promise<string|null> {
  try {
    const res = await axios.get('https://mindtrip.site/api/htp/v1/member/htp-last-result', {
      headers: {
        Authorization: token
      }
    })
    return res.data
  } catch(err) {
    console.log(err)
    return null
  }  
}


// 마이페이지에서 결과(사진) 조회하기
async function loadPictureResult(token:string) :Promise<pictureResultType[]|null> {
  try {
    const res = await axios.get('https://mindtrip.site/api/htp/v1/member/htp-results', {
      headers : {
        Authorization: token
      }
    })
    return res.data
  } catch(err) {
    console.log(err)
    return null
  }
}



export { getSurvey, getResult1, getResult2, changeLang, loadRecentResult, loadPictureResult }
