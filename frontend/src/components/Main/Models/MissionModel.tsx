import * as THREE from 'three'
import { useGLTF } from '@react-three/drei'
import { GLTF } from 'three-stdlib'

type GLTFResult = GLTF & {
  nodes: {
    stall_001: THREE.Mesh
  }
  materials: {
    ['Material.002']: THREE.MeshPhysicalMaterial
  }
}

export function MissionModel(props: JSX.IntrinsicElements['group']) {
  const { nodes, materials } = useGLTF('/models/mission.glb') as GLTFResult
  return (
    <group {...props} dispose={null}>
      <mesh
        castShadow
        receiveShadow
        geometry={nodes.stall_001.geometry}
        material={materials['Material.002']}
        position={[0.363, -2.9, 12.769]}
        rotation={[-0.036, -1.244, 0]}
      />
    </group>
  )
}

useGLTF.preload('/models/mission.glb')