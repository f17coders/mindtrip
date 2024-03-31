import { combineReducers, configureStore } from '@reduxjs/toolkit'
import chatSlice from './chatSlice';
import {consultSlice} from './consultSlice';
import { htpAnswer } from './htpSlice';
import {memberSlice, accessToken} from './memberSlice';
import { persistStore, persistReducer } from 'redux-persist';
import storageSession from 'redux-persist/lib/storage/session';



// storage 저장용
const persistConfig = {
  key: 'root',
  storage: storageSession,
  // 로컬에 저장하고 싶은 애만 빼주기
  whitelist: ['chat', 'member', 'accessToken', 'consultSlice', 'htpAnswer']
}

const persistedReducer = persistReducer(
  persistConfig,
  combineReducers({
    chat: chatSlice.reducer,
    consultSlice: consultSlice.reducer,
    member: memberSlice.reducer,
    accessToken: accessToken.reducer,
    htpAnswer: htpAnswer.reducer
  })
)


// 스토어 생성
const store = configureStore({
  reducer: persistedReducer,
  middleware: getDefaultMiddleware => getDefaultMiddleware({serializableCheck:false})
});

// 스토어의 타입을 내보내기 위한 유틸리티 타입
export type AppDispatch = typeof store.dispatch;
export type AppStore = typeof store;

// 여기서 RootState 타입을 정의, export
export type RootState = ReturnType<typeof store.getState>;

// persist 내보내기
export const persistor = persistStore(store)

// 스토어 내보내기
export default store;