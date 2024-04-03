import * as THREE from 'three'
import { useGLTF } from '@react-three/drei'
import { GLTF } from 'three-stdlib'

type GLTFResult = GLTF & {
  nodes: {
    Mesh011: THREE.Mesh
    Mesh011_1: THREE.Mesh
    Mesh011_2: THREE.Mesh
    Mesh011_3: THREE.Mesh
    Mesh011_4: THREE.Mesh
    Mesh011_5: THREE.Mesh
    Mesh011_6: THREE.Mesh
    Mesh011_7: THREE.Mesh
    Mesh011_8: THREE.Mesh
    Mesh011_9: THREE.Mesh
    Mesh011_10: THREE.Mesh
    Mesh011_11: THREE.Mesh
    Mesh011_12: THREE.Mesh
    Mesh011_13: THREE.Mesh
    Mesh011_14: THREE.Mesh
    Mesh011_15: THREE.Mesh
    Mesh011_16: THREE.Mesh
  }
  materials: {
    ['Material.020']: THREE.MeshPhysicalMaterial
    ['Material.001']: THREE.MeshStandardMaterial
    ['Material.006']: THREE.MeshStandardMaterial
    ['Material.014']: THREE.MeshPhysicalMaterial
    ['Material.003']: THREE.MeshPhysicalMaterial
    ['Material.010']: THREE.MeshPhysicalMaterial
    ['Material.011']: THREE.MeshPhysicalMaterial
    ['Material.012']: THREE.MeshPhysicalMaterial
    ['Material.013']: THREE.MeshPhysicalMaterial
    ['Material.015']: THREE.MeshPhysicalMaterial
    ['Material.016']: THREE.MeshPhysicalMaterial
    ['Material.017']: THREE.MeshPhysicalMaterial
    ['Material.018']: THREE.MeshPhysicalMaterial
    ['Material.019']: THREE.MeshPhysicalMaterial
    ['Material.021']: THREE.MeshPhysicalMaterial
    ['Material.022']: THREE.MeshPhysicalMaterial
    ['Material.023']: THREE.MeshPhysicalMaterial
  }
}

export function BaseModel(props: JSX.IntrinsicElements['group']) {
  const { nodes, materials } = useGLTF('./models/base.glb') as GLTFResult
  return (
    <group {...props} dispose={null}>
      <group
        position={[7.77, -3, 3.24]}
        rotation={[-1.652, -0.113, -0.154]}
        scale={[-0.274, -0.441, -0.419]}>
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011.geometry}
          material={materials['Material.020']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_1.geometry}
          material={materials['Material.001']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_2.geometry}
          material={materials['Material.006']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_3.geometry}
          material={materials['Material.014']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_4.geometry}
          material={materials['Material.003']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_5.geometry}
          material={materials['Material.010']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_6.geometry}
          material={materials['Material.011']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_7.geometry}
          material={materials['Material.012']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_8.geometry}
          material={materials['Material.013']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_9.geometry}
          material={materials['Material.015']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_10.geometry}
          material={materials['Material.016']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_11.geometry}
          material={materials['Material.017']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_12.geometry}
          material={materials['Material.018']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_13.geometry}
          material={materials['Material.019']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_14.geometry}
          material={materials['Material.021']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_15.geometry}
          material={materials['Material.022']}
        />
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Mesh011_16.geometry}
          material={materials['Material.023']}
        />
      </group>
    </group>
  )
}

useGLTF.preload('./models/base.glb')