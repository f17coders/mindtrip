

import * as THREE from 'three'
import { useGLTF } from '@react-three/drei'
import { GLTF } from 'three-stdlib'

type GLTFResult = GLTF & {
  nodes: {
    Cube011: THREE.Mesh
    Cube011_1: THREE.Mesh
    Cube011_2: THREE.Mesh
  }
  materials: {
    ['Clouds.001']: THREE.MeshStandardMaterial
    Clouds: THREE.MeshStandardMaterial
    ['Clouds.002']: THREE.MeshStandardMaterial
  }
}

export function CloudModel(props: JSX.IntrinsicElements['group']) {
  const { nodes, materials } = useGLTF('/models/cloud.glb') as GLTFResult
  return (
    <group {...props} dispose={null}>
      <group position={[-1, 12.26, -12.046]} rotation={[-Math.PI, 1.322, -Math.PI]}>
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Cube011.geometry}
          material={materials['Clouds.001']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Cube011_1.geometry}
          material={materials.Clouds}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Cube011_2.geometry}
          material={materials['Clouds.002']}
        />
      </group>
    </group>
  )
}

useGLTF.preload('/models/cloud.glb')
