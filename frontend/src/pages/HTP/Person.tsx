import { useState, useRef, useEffect } from "react"
import { Button } from "@nextui-org/react";
import { useNavigate } from "react-router-dom";
import Swal from "sweetalert2";
import Draw from "../../components/HTP/Draw";
import axios from 'axios'
import { useDispatch, useSelector } from "react-redux";
import { savePersonAnswer } from "../../store/htpSlice";
import { RootState } from "../../store/store";

// props의 타입을 지정해주자

type propsType1 = {
  goSurvey: () => void
}
type survey = {
  quetion_id:number,
  content:string,
  choices:{
    choice_id:number,
    content:string
  }[]
}
type propsType = {
  goNext: () => void
  survey: survey
  isLast: boolean
}

// 사람 그림 그리고, 설문조사가 나오는 페이지
function Person() {
  const dispatch = useDispatch()
  const [order, setOrder] = useState<number>(0)
  const goNext: () => void = () => {
    setOrder(order + 1);
  };

  const [isSurvey, setIsSurvey] = useState<boolean>(false)
  const goSurvey = function (): void {
    setIsSurvey(true)
  }

  // 처음에 질문 불러오기
  const [surveyList, setSurveyList] = useState<survey[]|null>(null)
  async function loadSurvey(): Promise<survey[]|null>{
    try{
      const res = await axios.get('https://mindtrip.site/api/htp/v0/question/person')
      return res.data
    } catch(err) {
      console.log(err)
      return null
    }
  }

  
  useEffect(() => {
    const fetchSurvey = async () => {
      try {
        let tempSurveys :survey[]|null = await loadSurvey() 
        tempSurveys ? setSurveyList(tempSurveys) : console.log('데이터가 없습니다!')
      } catch (err) {
        console.log(err)
      }
    }
    fetchSurvey()

    // 처음에 들어오면 리덕스 초기화
    dispatch(savePersonAnswer([]))
  } ,[])


  return (
    <div className=''>
      {
        isSurvey === true ? (<div>{
          surveyList?.map((survey, idx) => {
            if (order === idx) {
              if (idx === surveyList.length - 1) {
                return (<PersonSurvey goNext={goNext} survey={survey} isLast={true} key={idx}/>)
              } else {
                return (<PersonSurvey goNext={goNext} survey={survey} isLast={false} key={idx} />)
              }
            }
          })
        }
        {
          surveyList === null && <p>로딩중</p>
        }
        </div>) : (<PersonDraw goSurvey={goSurvey} />)
      }
    </div>
  )
}
export default Person

function PersonDraw({ goSurvey }: propsType1) {
  let tempAuthorization = useSelector((state: RootState) => state.accessToken)

  // 파일 제어용
  const [file, setFile] = useState<File | null>(null)
  const fileInput = useRef<HTMLInputElement>(null);
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    // 선택한 파일 정보를 저장
    const files = event.target.files
    if (files && files[0]) {
      setFile(files[0])
    }
  };
  
  const sendFile = async (data: FormData) => {
    await axios.post('https://mindtrip.site/api/htp/v1/test/person', data, {
      headers: {
        tempAuthorization: tempAuthorization,
        "Content-Type": "multipart/form-data"
      }
    })
  }
  const handleFile = async (file: FormData) => {
    await (sendFile(file))
    Swal.fire({
      title: '업로드완료',
      icon: "success"
    }).then(() => { goSurvey() })
  }

  // 파일 써먹을려면
  useEffect(() => {
    if (file != null) {
      let formData = new FormData()
      formData.append('file', file)
      for (const value of formData.values()) {
        console.log(value);
      }
      handleFile(formData)
    }
  }, [file])



  return (
    <div className="flex h-svh w-svh justify-center items-center flex-col">
      <p className="text-center mb-8 font-bold text-3xl">사람을 그려주세요.</p>
      <div className="relative border-2 rounded h-2/3 lg:w-2/3 w-full bg-white">
        <Draw />
        <Button
          className={`absolute bottom-0 right-0 m-3 opacity-50`}
          onClick={goSurvey}
        >다 그렸어요</Button>
      </div>
      <div className="flex items-center text-slate-500 mt-2">
        <p className="mr-3">그리기 힘들다면?</p>
        {/* 업로드 아이콘 */}
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.233-2.33 3 3 0 0 1 3.758 3.848A3.752 3.752 0 0 1 18 19.5H6.75Z" />
        </svg>
        <button onClick={() => { fileInput.current?.click() }}>
          <p className="underline ml-1 hover:text-violet-500 hover:cursor-pointer">업로드하기</p>
        </button>
      </div>
      {/* png형식의 파일만 올릴 수 있도록 함 */}
      <input type="file" accept=".png" ref={fileInput} onChange={handleChange} style={{ display: 'none' }} />
    </div>
  )
}

function PersonSurvey({ goNext, survey, isLast }: propsType) {
  const navigate = useNavigate()
  const dispatch = useDispatch()
  let person = useSelector((state:RootState) => state.htpAnswer.person)
  let answer = useSelector((state:RootState) => state.htpAnswer)
  let accessToken = useSelector((state:RootState) => state.accessToken) 

  const sendAnswer = async () => {
    const res = await axios.post('https://mindtrip.site/api/htp/v1/answer', {
      answer: {
        house: answer.house,
        tree: answer.tree,
        person: answer.person
      }
    }, {
      headers : {
        tempAuthorization: accessToken
      }
    })
    console.log(res)
  }

  const handleClick = async (questionId:number, answerId:number) => {
    if (person === null) {
      dispatch(savePersonAnswer([{question_id: questionId, choice_id: answerId}]))
    } else {
      dispatch(savePersonAnswer([...person, {question_id: questionId, choice_id: answerId}]))
    }

    if (isLast) {
      await sendAnswer()
      navigate('/htp/result')
    } else {
      goNext()
    }
  }
  
  return (
    <div className="flex h-svh w-svh justify-center items-center flex-col">
      <p className="text-center mb-5 font-bold text-2xl">{survey.content}</p>
      {
        survey.choices.map((choice, idx) => {
          return (
            <Button
              key={idx}
              variant="bordered"
              className='w-[90vw] lg:w-3/5 m-3 h-[10vh] px-3 text-md bg-white hover:bg-sky-800 hover:text-white shadow'
              onClick={() => {handleClick(survey.quetion_id, choice.choice_id)}}
            >
              {choice.content}
            </Button>
            )
        })
      }
    </div>
  )
}