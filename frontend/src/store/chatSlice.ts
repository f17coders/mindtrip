import { createSlice, PayloadAction } from '@reduxjs/toolkit'

// 채팅 관리용 변수들
// 1. 타입 정의
type ChatState = {
  isOpen: boolean
  isList: boolean
  selectedId: string | null, 
  isMine: boolean
}

// 2. 초기 상태 정의
const initialChatState :ChatState = {
  isOpen: false,
  isList: true,
  selectedId: null,
  isMine: true
}

// 3. 슬라이스 정의
const chatSlice = createSlice({
  name: 'chat',
  initialState: initialChatState,
  reducers: {
    // 채팅 여닫기 제어
    toggleOpen(state) {
      state.isOpen = !state.isOpen
    },
    // 보이는게 리스트인지(true) / 하나의 채팅방인지(false)
    changeList(state, action: PayloadAction<boolean>) {
      state.isList = action.payload
    },
    // 채팅방 아이디 넣으면 바꿔주는 함수
    changeSelectedId(state, action: PayloadAction<string| null>) {
      state.selectedId = action.payload
    },
    setisMine(state, action:PayloadAction<boolean>){
      state.isMine = action.payload
    }
  }
})

// 4. 함수들 내보내기
export const { toggleOpen, changeList, changeSelectedId, setisMine } = chatSlice.actions;


export default chatSlice