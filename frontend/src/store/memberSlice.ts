import { createSlice, PayloadAction } from '@reduxjs/toolkit'
import { memberType } from '../types/DataTypes';
// 회원용 변수들

// 토큰
type tokenType = {
  value:string
}

const initialToken:tokenType = {
  value: ''
}

const accessToken = createSlice({
  name: 'accessToken',
  initialState: initialToken,
  reducers: {
    // 토큰 저장
    saveToken(state, action:PayloadAction<string>) {
      return {
        ...state,
        value: action.payload
      }
    },
    // 토큰 삭제
    deleteToken(state) {
      return {
        ...state,
        value: ''
      }
    }
  }
})
export const {saveToken, deleteToken} = accessToken.actions;

// 회원 정보
// 1. 타입정의 -> 타입 파일에
// 2. 초기 상태 정의
const initialMemberState :memberType = {
  memberId: null,
  nickname: null,
  socialId: null,
  villageId: null,
  villageName: 'blueberry',
  level: 0,
  missionCount: 0,
  accessToken: null
}

// 3. 슬라이스 정의
const memberSlice = createSlice({
  name: 'member',
  initialState: initialMemberState,
  reducers: {
    // 사용자 정보 저장
    saveUserInfo(state, action: PayloadAction<memberType>) {
      return({
        ...state,
        ...action.payload
      })
    },
    saveVillage(state, action: PayloadAction<'apple'|'orange'|'pineapple'|'watermelon'|'grape'|'peach'|'blueberry'|'kakao'>) {
      return({
        ...state,
        villageName: action.payload
      })
    },
    countUpdate(state){
      const nowLevel = state.level
      const newCount = state.missionCount + 1
      if (newCount === 3) {
        return({
          ...state,
          level: nowLevel + 1,
          missionCount: newCount
        })
      } else if (newCount === 9) {
        return({
          ...state,
          level: nowLevel + 1,
          missionCount: newCount
        })
      } else if (newCount === 15) {
        return({
          ...state,
          level: nowLevel + 1,
          missionCount: newCount
        })
      } else if (newCount === 21) {
        return({
          ...state,
          level: nowLevel + 1,
          missionCount: newCount
        })
      } else if (newCount === 30) {
        return({
          ...state,
          level: nowLevel + 1,
          missionCount: newCount
        })
      } else {
        return({
          ...state,
          missionCount: newCount
        })
      }
    },
    deleteUserInfo(state) {
      return {
        ...state,
        memberId: null,
        nickname: null,
        socialId: null,
        villageId: null,
        villageName: 'blueberry',
        level: 0,
        missionCount: 0,
        accessToken: null
      }
    }
  }
})

// 4. 함수들 내보내기
export const {saveUserInfo, saveVillage, countUpdate, deleteUserInfo} = memberSlice.actions;


export {memberSlice, accessToken}