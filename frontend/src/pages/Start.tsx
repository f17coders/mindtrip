import { Link, useNavigate } from "react-router-dom"
import Background from '../components/HTP/BackGround'
import { useDispatch } from "react-redux"
import { saveToken } from "../store/memberSlice"
import axios from "axios"
import logo from './../assets/logo.png'

function Start() {
  const navigate = useNavigate()
  const dispatch = useDispatch()

  const getTempToken = async () => {
    // console.log('테스트')
    try {
      const res = await axios.get('https://mindtrip.site/api/htp/v0/temp_token')
      dispatch(saveToken(res.data.Authorization))
    } catch(err) {
      console.log(err)
    }
    
  }

  const handleClick = async () => {
    await getTempToken()
    navigate('/htp/house')
  }

  return (
    <div className="relative">
      <div className="absolute bottom-2 right-2 z-10 text-[rgba(192,192,192,0.8)] text-sm" style={{fontFamily:'JamsilThin'}}>저희 서비스는 앱 환경에 최적화되어 있습니다.</div>
      <div className="flex items-center justify-center h-screen w-screen relative z-10">
        <div className="rounded-xl text-center mb-[25vh] bg-[rgba(255,255,255,0.7)] md:bg-transparent p-7 md:w-[50%] md:h-[40%]">
          <img src={logo} alt="logo_img" className="w-[100px] sm:w-[130px] mx-auto mb-3"/>
          <p className="font-bold text-xl leading-relaxed md:text-3xl">HTP 검사를 통해<br />나를 위한 여정을 떠나보세요.</p>
          <div className='mt-6 my-2' onClick={handleClick}>
            <MyBtn />
          </div>
          <p className="mt-3 text-sm text-slate-400">이미 검사를 진행하셨다면,<br />
            <Link to={'/login'} className="underline text-cyan-500">로그인 해주세요.</Link>
          </p>
        </div>
      </div>
      <div className="absolute top-0 left-0">
        <Background />
      </div>
    </div>
  )
}

export default Start


// 불러온 버튼
function MyBtn() {
  return (
    <button
      className="relative py-2 px-8 text-black text-base font-bold nded-full overflow-hidden bg-white rounded-full transition-all duration-400 ease-in-out shadow-md hover:scale-105 hover:text-white hover:shadow-lg active:scale-90 before:absolute before:top-0 before:-left-full before:w-full before:h-full before:bg-gradient-to-r before:from-blue-500 before:to-blue-300 before:transition-all before:duration-500 before:ease-in-out before:z-[-1] before:rounded-full hover:before:left-0"
    >
      <p className="flex text-sm md:text-xl" style={{fontFamily:'JamsilThin'}}>검사하러가기<StartIcon /></p>
    </button>
  )
}

// 버튼에 들어갈 아이콘 컴포넌트
function StartIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 ml-2">
      <path fillRule="evenodd" d="M9.315 7.584C12.195 3.883 16.695 1.5 21.75 1.5a.75.75 0 0 1 .75.75c0 5.056-2.383 9.555-6.084 12.436A6.75 6.75 0 0 1 9.75 22.5a.75.75 0 0 1-.75-.75v-4.131A15.838 15.838 0 0 1 6.382 15H2.25a.75.75 0 0 1-.75-.75 6.75 6.75 0 0 1 7.815-6.666ZM15 6.75a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5Z" clipRule="evenodd" />
      <path d="M5.26 17.242a.75.75 0 1 0-.897-1.203 5.243 5.243 0 0 0-2.05 5.022.75.75 0 0 0 .625.627 5.243 5.243 0 0 0 5.022-2.051.75.75 0 1 0-1.202-.897 3.744 3.744 0 0 1-3.008 1.51c0-1.23.592-2.323 1.51-3.008Z" />
    </svg>
  )
}