import axios from "axios";
import { memberType } from "../types/DataTypes";
import Swal from "sweetalert2";

// 로그인하기

type memberInfo = {
  memberId: number,
  token: string
}

async function login(id: string, password: string): Promise<memberInfo | void> {
  try {
    const res = await axios.post('https://mindtrip.site/api/members/v0/login', {
      'id': id,
      'password': password
    });
    return res.data.result
  } catch (err) {
    console.log(err);
  }
}



// 유저정보 로딩하기
async function loadUser(memberId :number): Promise<memberType | void> {
  try {
    const res = await axios.get(`https://mindtrip.site/api/members/v0/${memberId}`)
    return res.data.result
    // 여기서 멤버 정보를 return 해줘야함
  } catch (err) {
    console.log(err)
  }
}


// 회원가입하기
async function signup(id: string, password: string, nickname: string): Promise<number | null> {
  try {
    const res = await axios.post('https://mindtrip.site/api/members/v0/register', {
      'id': id,
      'password': password,
      'nickname': nickname
    });
    console.log(res)
    return res.data.result
  } catch (err) {
    console.log(err);
    return null
  }
}


// 검사결과 등록하기
async function registerResult(memberId: number, tempToken: string): Promise<boolean> {
  console.log('등록시도하러감')
  try {
    const res = await axios.post('https://mindtrip.site/api/htp/v1/register', {
      'member_id': memberId.toString()
    }, {
      headers: {
        Authorization: tempToken
      }
    })
    // 메세지가 있는건 성공
    console.log(res)
    console.log(res.data)
    if (res.data.message) {
      return true
    }
    if (res.data.detail) {
      Swal.fire({
        text: res.data.detail
      })
      return false
    }
    return false
  } catch (err) {
    console.log(err)
    return false
  }
}

export { login, signup, loadUser, registerResult }