import {Input,Button} from "@nextui-org/react";
import ResignIcon from '../../../atoms/Icons/ResignIcon';
import kakao from '../../../assets/login/kakao.png'
import { RootState } from "../../../store/store";
import { useSelector } from "react-redux";
import { villageBackgroundColor } from "../../../atoms/color";



function MypageFix() {
  let member = useSelector((state:RootState) => state.member)
  
  return (
    <div className={`${villageBackgroundColor[member.villageName]} flex flex-col justify-around items-center min-h-[80vh] relative`}>
    <div className="flex  ">
      <div className="bg-white shadow-medium rounded-large rounded-lg h-[70vh] w-[70vw]  ">
      <Input className="my-16 max-w-[50vw] mx-auto" type="email" variant= "underlined" label="이름" placeholder="이름이름" isDisabled />
      <Input className="my-16 max-w-[50vw] mx-auto" type="email" variant= "underlined" label="닉네임" placeholder="원래 닉네임" />
      <div className="flex justify-center">
        <Button className=" mx-auto my-2 mt-4 bg-[#FEE500] pr-7 shadow " size='lg'>
          <img className='w-8 h-9 mb-1' src={kakao} alt="kakaoLogo" />
          <p>카카오로 연동하기</p>
        </Button>
      </div>
      <div className="mx-auto my-8 flex justify-center">
        <Button size="lg" color="danger" variant="bordered" startContent={<ResignIcon/>}>
        회원 탈퇴하기
      </Button>
      </div>
      </div>
    </div>
    </div>
  );
  
  //   <div className="w-full flex flex-col gap-4">
  //       <div key= "underlined" className="flex w-full flex-wrap md:flex-nowrap mb-6 md:mb-0 gap-4">
  //         <Input type="email" variant= "underlined" label="Email" />
  //         <Input type="email" variant= "underlined" label="Email" placeholder="Enter your email" />
  //       </div>
  //   </div>  
  // );
}
export default MypageFix
