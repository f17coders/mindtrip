import Start from './pages/Start';
import Main from './pages/Main';
import Signup from './pages/Signup';
import Login from './pages/Login';
import Consult from './pages/Consult/Consult';
import ConsultOther from './pages/Consult/ConsultOther';
import ConsultShared from './pages/Consult/ConsultShared';
import HTP from './pages/HTP/HTP';
import House from './pages/HTP/House';
import Tree from './pages/HTP/Tree';
import Person from './pages/HTP/Person';
import Result from './pages/HTP/Result';
import Mission from './pages/Mission/Mission';
import Fixmission from './pages/Mission/FixMission'
import Postit from './pages/Postit/Postit';
import Mypage from './pages/Mypage/Mypage';
import MypageHTP from './pages/Mypage/MypageDetail/MypageHTP';
import MypageProgress from './pages/Mypage/MypageDetail/MypageProgress';
import MypageHistory from './pages/Mypage/MypageDetail/MypageHistory';
import { Routes, Route } from 'react-router-dom'
import { ProtectedRoute } from './services/protectRoute';
import { initializeApp } from 'firebase/app';
import { getMessaging, getToken } from 'firebase/messaging';
import { saveNotificationToken } from './store/notificationSlice';
import { useDispatch } from 'react-redux';
import { RootState } from './store/store';
import { useSelector } from 'react-redux';

function App() {

  // 앱 시작하면 noti용 토큰을 저장해둠
  const dispatch = useDispatch()
  let notificationToken = useSelector((state:RootState) => state.notificationToken.value)

  const firebaseConfig = ({
    apiKey: import.meta.env.VITE_APP_API_KEY,
    authDomain: import.meta.env.VITE_APP_AUTH_DOMAIN,
    projectId: import.meta.env.VITE_APP_PROJECT_ID,
    storageBucket: import.meta.env.VITE_APP_STORAGE_BUCKET,
    messagingSenderId: import.meta.env.VITE_APP_MESSAGING_SENDER_ID,
    appId: import.meta.env.VITE_APP_APP_ID,
    measurementId: import.meta.env.VITE_APP_MEASUREMENT_ID,
  });

  initializeApp(firebaseConfig)
  const messaging = getMessaging()

  getToken(messaging, {vapidKey: import.meta.env.VITE_APP_VAPID_KEY}). then((currentToken) => {
    if (currentToken) {
      if (notificationToken === '') {
        dispatch(saveNotificationToken(currentToken))
      }
      // console.log('토큰임', currentToken)
    } else {
      console.log('토큰 못가져옴')
    }
  }).catch((err) => console.log(err))


  return (
    <div className="App select-none">
      <Routes>
        <Route element={<ProtectedRoute />}>
          <Route path='/consult' Component={Consult}/>
          <Route path='/consult/other' Component={ConsultOther} />
          <Route path='/consult/shared' Component={ConsultShared} />
          <Route path='/mission' Component={Mission}/>
          <Route path='/fixmission' Component={Fixmission}/>
          <Route path='/postit' Component={Postit}/>
          <Route path='/mypage' Component={Mypage}>
          <Route index Component={MypageHTP}/>
          <Route path='htp' Component={MypageHTP}/>
          <Route path='history' Component={MypageHistory}/>
          <Route path='progress' Component={MypageProgress}/>
        </Route>

        </Route>
        <Route path='/' Component={Start}/>
        <Route path='/main' Component={Main}/>
        <Route path='/signup' Component={Signup}/>
        <Route path='/login' Component={Login}/>
        <Route path='/htp' Component={HTP}>
          <Route path='house' Component={House}/>
          <Route path='tree' Component={Tree}/>
          <Route path='person' Component={Person}/>
          <Route path='result' Component={Result}/>
        </Route>
        
        <Route path='*' element={<div>없는 페이지에요</div>}/>
      </Routes>
    </div>
  );
}

export default App;

