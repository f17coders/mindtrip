import * as THREE from 'three'
import { useRef, useEffect } from 'react'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { useNavigate } from 'react-router-dom'

type propsType = {
    go:string|null
}

function MainBackground({go} :propsType) {
  const navigate = useNavigate()

  // 3d 모델
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const loader = new GLTFLoader()

  // 배경
  const scene = new THREE.Scene()

  // 카메라
  const camera = new THREE.PerspectiveCamera(10, window.innerWidth / window.innerHeight, 10, 200)
  camera.position.set(5, 45, 150)
  camera.rotation.x -= 0.25

  // 조명
  const directionalLight = new THREE.DirectionalLight('#faf3ea', 2.5); // color, intensity
  directionalLight.position.set(0, 100, 100); // x, y, z
  scene.add(directionalLight);

  useEffect(() => {
    if (canvasRef.current) {
      // 렌더
      let renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current })

      // 윈도우 사이즈 바껴도 계속 맞춰주기
      window.addEventListener('resize', () => {
        renderer.setSize(window.innerWidth, window.innerHeight)
        // 메쉬비율 유지
        camera.aspect = window.innerWidth / window.innerHeight
        camera.updateProjectionMatrix()
        renderer.render(scene, camera)
      })

      // 배경(투명하게 설정)
      renderer.setClearColor(0x000000, 0);
      renderer.setSize(window.innerWidth, window.innerHeight)

      // 모델 불러오기
      loader.load('/village.glb', function (gltf: any) {
        scene.add(gltf.scene)

        // 랜더링
        renderer.render(scene, camera)

        // 카메라를 움직여보자
        function animateCamera(startPosition:THREE.Vector3 ,targetPosition:THREE.Vector3, duration:number, callback:Function|null) :void {
          const startTime:number = performance.now()
          function update() {
            const elapsedTime:number = performance.now() - startTime
            const progress:number = elapsedTime / duration
            if (progress < 1) {
              // 처음위치, 타겟위치, 얼마나 갈껀지
              camera.position.lerpVectors(startPosition, targetPosition, progress);
              requestAnimationFrame(update);
            } else {
              camera.position.copy(targetPosition);
              if (callback) {callback()} else {
                navigate(`/${go}`)
              }
            }
            renderer.render(scene, camera);
          }
          update()
        }
        
        if (go === 'mission') {
          animateCamera(new THREE.Vector3(5, 45, 150), new THREE.Vector3(3, 25, 80), 500, () => {
          animateCamera(new THREE.Vector3(3, 25, 80), new THREE.Vector3(2.5, 14, 50), 500, null)})
        }
        if (go === 'consult') {
          animateCamera(new THREE.Vector3(5, 45, 150), new THREE.Vector3(2, 25, 80), 500, () => {
          animateCamera(new THREE.Vector3(2, 25, 80), new THREE.Vector3(0.7, 17, 45), 500, null)})
        }
        if (go === 'postit') {
          animateCamera(new THREE.Vector3(5, 45, 150), new THREE.Vector3(3, 25, 80), 500, () => {
          animateCamera(new THREE.Vector3(3, 25, 80), new THREE.Vector3(5, 17, 50), 500, null)})
        }
        if (go === 'mypage') {
          animateCamera(new THREE.Vector3(5, 45, 150), new THREE.Vector3(5, 25, 80), 500, () => {
          animateCamera(new THREE.Vector3(5, 25, 80), new THREE.Vector3(8, 17, 60), 500, null)})
        }
      }, undefined, function (err: any) {
        console.log(err)
      })
    }
  }, [go])

  return (
    <div className='bg-sky-100'>
      <canvas ref={canvasRef} className='w-screen h-screen'></canvas>
    </div>
  )
}

export default MainBackground