import { createSlice, PayloadAction } from '@reduxjs/toolkit'
// 검사용 변수들

// 타입지정
type eleType = {
    question_id: number,
    choice_id: number
}

type answerType = {
    house: eleType[]|null,
    tree: eleType[]|null,
    person: eleType[]|null
}

// 초기 상태 지정
const initialAnswer: answerType = {
    house: null,
    tree: null,
    person: null
}

const htpAnswer = createSlice({
    name: 'htpAnswer',
    initialState: initialAnswer,
    reducers: {
        // 집 저장
        saveHouseAnswer (state, action: PayloadAction<eleType[]>) {
            return {
                ...state,
                house: action.payload
            }
        },
        // 나무 저장
        saveTreeAnswer(state, action: PayloadAction<eleType[]>) {
           return {
            ...state,
            tree: action.payload
           }
        },
        // 사람 저장
        savePersonAnswer(state, action: PayloadAction<eleType[]>) {
          return {
            ...state,
            person: action.payload
          }
        }
    }
})
export const { saveHouseAnswer,saveTreeAnswer,savePersonAnswer } = htpAnswer.actions;

export { htpAnswer }