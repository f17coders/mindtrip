import Tree from "../components/Loading/MissionTree"


function Loading() {
  return (
    <div>
      <div className="w-full h-full z-10 bg-[rgba(192,192,192,0.5)] absolute top-0 left-0">
        <div className="mt-[40vh]">
          <Tree />
          <p className="text-center text-xl">결과 분석 중, 잠시만 기다려 주세요</p>
        </div>
      </div>
    </div>
  )
}

export default Loading