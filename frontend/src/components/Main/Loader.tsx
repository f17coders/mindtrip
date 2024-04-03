import { Html, useProgress } from '@react-three/drei'

function Loader() {
  const { progress } = useProgress()
  return <Html center>{Math.round(progress)}%</Html>
}

export default Loader