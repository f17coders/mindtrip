
import * as THREE from 'three'
import { useGLTF } from '@react-three/drei'
import { GLTF } from 'three-stdlib'

type GLTFResult = GLTF & {
  nodes: {
    Cube194: THREE.Mesh
    Cube194_1: THREE.Mesh
    Cube194_2: THREE.Mesh
  }
  materials: {
    ['CartoonTown_01.007']: THREE.MeshPhysicalMaterial
    ['Material.008']: THREE.MeshStandardMaterial
    ['Material.007']: THREE.MeshStandardMaterial
  }
}

export function PostitModel(props: JSX.IntrinsicElements['group']) {
  const { nodes, materials } = useGLTF('/models/postit.glb') as GLTFResult
  return (
    <group {...props} dispose={null}>
      <group
        position={[3, -2.9, 5.48]}
        rotation={[Math.PI / 2, 0, 0.17]}
        scale={[0.02, 0.01, 0.02]}>
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Cube194.geometry}
          material={materials['CartoonTown_01.007']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Cube194_1.geometry}
          material={materials['Material.008']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Cube194_2.geometry}
          material={materials['Material.007']}
        />
      </group>
    </group>
  )
}

useGLTF.preload('/models/postit.glb')