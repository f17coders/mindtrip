import { Suspense } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Environment } from "@react-three/drei"
import { BaseModel } from "./Models/BaseModel"
import { MypageModel } from "./Models/MypageModel"
import { ConsultModel } from "./Models/ConsultModel"
import { MissionModel } from "./Models/MissionModel"
import { PostitModel } from "./Models/PostitModel"
import { CloudModel } from "./Models/CloudModel"
import { useNavigate } from "react-router-dom"
import Loader from "./Loader"
import { Sky } from "@react-three/drei"

function TestBackground() {
  const navigate = useNavigate()
  return (
    <Canvas linear camera={{ position: [30, 80, 100], fov: 15 }}>
      {/* <color attach="background" args={[198, 223, 249]}/> */}
      <Suspense fallback={<Loader/>}>
        <ambientLight intensity={4} />
          <BaseModel/>
          <MypageModel onClick={() => navigate('/mypage/htp')}/>
          <ConsultModel onClick={() => navigate('/consult')}/>
          <MissionModel onClick={() => navigate('/mission')}/>
          <PostitModel onClick={() => navigate('/postit')}/>
          <CloudModel/>
        <Environment preset="sunset" />
        <Sky/>
        <OrbitControls />
      </Suspense>
    </Canvas>
  )
}

export default TestBackground
