import * as THREE from 'three'
import { useGLTF } from '@react-three/drei'
import { GLTF } from 'three-stdlib'

type GLTFResult = GLTF & {
  nodes: {
    house_001: THREE.Mesh
  }
  materials: {
    ['Material.009']: THREE.MeshPhysicalMaterial
  }
}

export function ConsultModel(props: JSX.IntrinsicElements['group']) {
  const { nodes, materials } = useGLTF('./models/consult.glb') as GLTFResult
  return (
    <group {...props} dispose={null}>
      <mesh
        castShadow
        receiveShadow
        geometry={nodes.house_001.geometry}
        material={materials['Material.009']}
        position={[0, -2.7, -1.398]}
        rotation={[0.141, -1.188, 0]}
        scale={[1.85, 1.287, 1.464]}
      />
    </group>
  )
}

useGLTF.preload('./models/consult.glb')