import Swal from "sweetalert2";
import { useNavigate } from "react-router-dom";
import { getResult1, getResult2, changeLang } from "../../api/htp";
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";
import { saveVillage } from "../../store/memberSlice";
import { useDispatch } from "react-redux";
import { villageNameType } from "../../types/DataTypes";
import { villageBackgroundColor, villageTextColor } from "../../atoms/color";


function Result() {
  let tempToken = useSelector((state: RootState) => state.accessToken.value)
  const dispatch = useDispatch()

  const [result1, setResult1] = useState<string>('')
  const [result2, setResult2] = useState<villageNameType>('apple')

  // 결과
  const fetchResult1 = async () => {
    try {
      let tempResult = await getResult1(tempToken)
      if (tempResult) {
        setResult1(tempResult)
      }
    } catch (err) {
      console.log(err)
    }
  }

  // 마을
  const fetchResult2 = async () => {
    try {
      let tempResult:villageNameType|null = await getResult2(tempToken)
      console.log(tempResult)
      if (tempResult != null) {
        setResult2(tempResult)
        dispatch(saveVillage(tempResult))
      }
    } catch (err) {
      console.log(err)
    }
  }


  useEffect(() => {
   // 처음 검사결과 가져오기
   fetchResult1()
   fetchResult2()
  }, [])


  return(
  <div>
      <p className="text-center text-4xl pt-8 pb-8">검사 결과</p>
      <div className={`
        ${villageBackgroundColor[result2]} 
        rounded mx-auto mb-5 shadow
        w-[80vw] lg:w-[50vw] min-h-[12vh]
        flex items-center pl-5`
      }>
        <img src={`/Image/${result2}.png`} className="w-[30%] mx-3 mt-2"/>
        <p className="text-center">당신의 마을은<br/><span className={villageTextColor[result2]}>{changeLang(result2)}</span>마을입니다!</p>
      </div>
      <div className="w-full lg:w-[70vw] mx-auto text-center border-3 rounded-md bg-white min-h-[45vh] p-7">
        <p className="min-h-[40vh] text-lg" style={{fontFamily: 'JamsilThin'}}>{result1}</p>
      </div>
    <MyBtn village={result2}/>
  </div>
  )
}

export default Result


type proptype = {
  village : villageNameType
}
function MyBtn({village}: proptype) {
  const navigate = useNavigate()

  let member = useSelector((state:RootState) => state.member)
  const goNextStep = () :void => {
    // 로그인이 되어있다면
    if (member.memberId) {
      navigate('/main')
    } else {
      Swal.fire({
        html:`
         <p>로그인이나 회원가입을 통해<br/>마을에 들어갈 수 있어요.</p>
        `,
        showCancelButton: true,
        confirmButtonText: '로그인하기',
        cancelButtonText: '회원가입하기'
      }).then((result)=>{
        if (result.isConfirmed) {
          navigate('/login')
        } else {
          navigate('/signup')
        }
      })
    }
  }

  return(
    <button
      className={`fixed bottom-10 left-1/2 translate-x-[-50%] py-2 px-8 h-14 w-48 text-black text-base font-bold nded-full overflow-hidden bg-white rounded-full transition-all duration-400 ease-in-out shadow-md hover:scale-105 hover:text-white hover:shadow-lg active:scale-90 before:absolute before:top-0 before:-left-full before:w-full before:h-full before:bg-gradient-to-r before:from-blue-500 before:to-blue-300 before:transition-all before:duration-500 before:ease-in-out before:z-[-1] before:rounded-full hover:before:left-0`}
      onClick={goNextStep}
    >
      <p><span className={`${villageTextColor[village]}`}>{changeLang(village)}</span>마을 가기</p>
    </button>
  )
}