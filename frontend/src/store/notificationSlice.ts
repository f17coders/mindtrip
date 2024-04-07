import { createSlice, PayloadAction } from '@reduxjs/toolkit'

// 알림용

// 2. 초기 상태 정의
const initialNotiToken :{value: string} = {value: ''}

// 3. 슬라이스 정의
const notificationToken = createSlice({
  name: 'notificationToken',
  initialState: initialNotiToken,
  reducers: {
    // 토큰 저장
    saveNotificationToken(state, action: PayloadAction<string>) {
      state.value = action.payload
    }
  }
})

// 4. 함수들 내보내기
export const { saveNotificationToken } = notificationToken.actions;


export {notificationToken}