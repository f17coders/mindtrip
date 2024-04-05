import * as THREE from 'three'
import { useGLTF } from '@react-three/drei'
import { GLTF } from 'three-stdlib'

type GLTFResult = GLTF & {
  nodes: {
    Mesh528: THREE.Mesh
    Mesh528_1: THREE.Mesh
  }
  materials: {
    ['CartoonTown_01.004']: THREE.MeshPhysicalMaterial
    ['Material.004']: THREE.MeshPhysicalMaterial
  }
}

export function MypageModel(props: JSX.IntrinsicElements['group']) {
  const { nodes, materials } = useGLTF('/models/mypage.glb') as GLTFResult
  return (
    <group {...props} dispose={null}>
      <group
        position={[8.617, -3, 14.706]}
        rotation={[Math.PI / 2, 0, -0.065]}
        scale={[0.013, 0.01, 0.018]}>
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh528.geometry}
          material={materials['CartoonTown_01.004']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh528_1.geometry}
          material={materials['Material.004']}
        />
      </group>
    </group>
  )
}

useGLTF.preload('/models/mypage.glb')