import { useGLTF } from '@react-three/drei'

const LazyModel = () => {
  const model = useGLTF('./village.glb')
  return <primitive object={model.scene}/>;
};

export default LazyModel;