import { Input, Card, CardBody, Button } from "@nextui-org/react";
import { useEffect, useState } from 'react';
import { loadUser, login } from "../api/member";
import { useDispatch } from "react-redux";
import { saveToken, saveUserInfo } from "../store/memberSlice";
import { useNavigate } from "react-router-dom";
import { memberType } from "../types/DataTypes";

type memberInfo = {
  memberId: number,
  token: string
}

function Login() {
  const dispatch = useDispatch()
  const navigate = useNavigate()

  // 비밀번호 보이고 안보이고 제어
  const [isVisible, setIsVisible] = useState(false);
  const toggleVisibility = () => setIsVisible(!isVisible);

  // 아이디
  const [id, setId] = useState<string>('')
  const [checkId, setCheckId] = useState<boolean>(true)

  // 비밀번호
  const [password, setPassword] = useState<string>('')
  const [checkPassword, setCheckPassword] = useState<boolean>(true)

  useEffect(() => {
    if (id != '') {
      setCheckId(true)
    } 
    if (password != '') {
      setCheckPassword(true)
    }
  }, [id, password])

  // 유저 정보 저장
  const saveUser = async function (memberId: number) {
    const userInfo: memberType | void = await loadUser(memberId)
    if (userInfo) {
      dispatch(saveUserInfo(userInfo))
    }
  }


  // 로그인 로직
  const handleLogin = async function () {
    if (id === '') {
      setCheckId(false)
    } else if (password === '') {
      setCheckPassword(false)
    } else {
      const tempInfo: memberInfo | void = await login(id, password)
      if (tempInfo != null) {
        dispatch(saveToken(tempInfo.token))
        saveUser(tempInfo.memberId)
        navigate('/main')
      }
    }
  }

  return (
    <Card
      className="w-full h-[90vh] mt-[5vh] sm:h-[90vh] sm:mt-[5vh] mx-auto sm:w-3/5 xl:w-1/3"
    >
      <CardBody className="flex-col content-center justify-around my-[15%] sm:my-[30%]">
        <p className="text-center text-4xl mb-12">로그인</p>
        {/* 아이디 입력 창 */}
        <Input
          isClearable
          value={id}
          onValueChange={setId}
          className="mx-auto my-5 w-[85%] md:w-[70%]"
          variant="underlined"
          label='아이디'
          placeholder="아이디를 입력해주세요."
          errorMessage={checkId? '':'아이디를 입력해주세요'}
        />
        {/* 비밀번호 입력 창 */}
        <Input
          label="비밀번호"
          value={password}
          onValueChange={setPassword}
          variant="underlined"
          placeholder="비밀번호를 입력해주세요."
          errorMessage={checkPassword? '':'비밀번호를 입력해주세요'}
          endContent={
            <button className="focus:outline-none" type="button" onClick={toggleVisibility}>
              {isVisible ? (
                <EyeSlashFilledIcon />
              ) : (
                <EyeFilledIcon />
              )}
            </button>
          }
          type={isVisible ? "text" : "password"}
          className="mx-auto my-5 w-[85%] md:w-[70%]"

        />
        {/* 로그인 버튼 */}
        <Button
          onClick={handleLogin}
          size='lg'
          className="w-[80%] mx-auto my-5 bg-[#eeeeee] shadow md:w-[80%]"
        >로그인</Button>
      </CardBody>
    </Card>
  )
}
export default Login


// 비밀번호용 아이콘
function EyeSlashFilledIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 mr-1 text-2xl text-default-400 pointer-events-none">
      <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
      <path fillRule="evenodd" d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 0 1 0-1.113ZM17.25 12a5.25 5.25 0 1 1-10.5 0 5.25 5.25 0 0 1 10.5 0Z" clipRule="evenodd" />
    </svg>
  )
}

function EyeFilledIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 mr-1 text-2xl text-default-400 pointer-events-none">
      <path d="M3.53 2.47a.75.75 0 0 0-1.06 1.06l18 18a.75.75 0 1 0 1.06-1.06l-18-18ZM22.676 12.553a11.249 11.249 0 0 1-2.631 4.31l-3.099-3.099a5.25 5.25 0 0 0-6.71-6.71L7.759 4.577a11.217 11.217 0 0 1 4.242-.827c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113Z" />
      <path d="M15.75 12c0 .18-.013.357-.037.53l-4.244-4.243A3.75 3.75 0 0 1 15.75 12ZM12.53 15.713l-4.243-4.244a3.75 3.75 0 0 0 4.244 4.243Z" />
      <path d="M6.75 12c0-.619.107-1.213.304-1.764l-3.1-3.1a11.25 11.25 0 0 0-2.63 4.31c-.12.362-.12.752 0 1.114 1.489 4.467 5.704 7.69 10.675 7.69 1.5 0 2.933-.294 4.242-.827l-2.477-2.477A5.25 5.25 0 0 1 6.75 12Z" />
    </svg>
  )
}