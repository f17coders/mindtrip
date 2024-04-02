import { Outlet } from "react-router-dom"
import { getSurvey } from "../../api/htp"
import { useEffect } from "react"
import { useDispatch } from "react-redux"
import { saveSurveys } from "../../store/htpSlice"

function HTP() {
  // 설문조사 가져와서 redux에 박기
  const dispatch = useDispatch()

  const fetchSurvey = async () => {
    const res = await getSurvey()
    if (res) {
      dispatch(saveSurveys(res))
    }
  }
  
  useEffect(() => {
    fetchSurvey()
  })

  return(
    <div className="md:mx-auto w-[100vw] h-screen bg-sky-100">
      <Outlet/>
    </div>
  )
}

export default HTP