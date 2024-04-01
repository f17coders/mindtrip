import Swal from "sweetalert2";
import { useNavigate } from "react-router-dom";
import { getResult1, getResult2, changeLang } from "../../api/htp";
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";

function Result() {
  let tempToken = useSelector((state: RootState) => state.accessToken)

  const [result1, setResult1] = useState<string>('')
  const [result2, setResult2] = useState<string>('')

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
      let tempResult = await getResult2(tempToken)
      console.log(tempResult)
      if (tempResult) {
        setResult2(tempResult)
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
  <div className="">
      <p className="text-center text-4xl py-10">검사 결과</p>
      <div className="w-full lg:w-[70vw] mx-auto text-center border-3 rounded-md bg-white min-h-[65vh] p-7">
        <p className="min-h-[55vh] text-lg" style={{fontFamily: 'JamsilThin'}}>{result1}</p>
      </div>
    <MyBtn village={result2}/>
  </div>
  )
}

export default Result

type propsType = {
  village :string
}
function MyBtn({village} :propsType) {
  const navigate = useNavigate()

  const checkVillage = () :void => {
    Swal.fire({
      text: `당신은 ${changeLang(village)} 마을에 도착했습니다.`
    }).then(() => {
      navigate('/main')
    })
  }

  return(
    <button
      className="fixed bottom-10 left-1/2 translate-x-[-50%] py-2 px-8 h-14 w-48 text-black text-base font-bold nded-full overflow-hidden bg-white rounded-full transition-all duration-400 ease-in-out shadow-md hover:scale-105 hover:text-white hover:shadow-lg active:scale-90 before:absolute before:top-0 before:-left-full before:w-full before:h-full before:bg-gradient-to-r before:from-blue-500 before:to-blue-300 before:transition-all before:duration-500 before:ease-in-out before:z-[-1] before:rounded-full hover:before:left-0"
      onClick={checkVillage}
    >
      <p>내 마을 확인하기</p>
    </button>
  )
}