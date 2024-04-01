import * as THREE from 'three'
import { useRef, useEffect } from 'react'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

function Background() {
  // 3d 모델
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const loader = new GLTFLoader()

  // 배경
  const scene = new THREE.Scene()

  // 카메라
  const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 100, 1000)
  camera.position.set(0, 400, 250)
  camera.rotation.x -= 0.1

  // 조명
  const directionalLight = new THREE.DirectionalLight(0xffffff, 4); // color, intensity
  directionalLight.position.set(0, 1, 2); // x, y, z
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
      loader.load('/sphere.glb', function (gltf: any) {
        scene.add(gltf.scene)

        // 랜더링
        renderer.render(scene, camera)

        // 물체를 움직여보자
        function animateScene() :void {
          function update() {
            requestAnimationFrame(update);
            gltf.scene.rotation.x += 0.002
            renderer.render(scene, camera);
          }
          update()
        }

        animateScene()
      }, undefined, function (err: any) {
        console.log(err)
      })
    }
  }, [])

  return (
    <div className='bg-sky-100'>
      <canvas ref={canvasRef} className='w-screen h-screen'></canvas>
    </div>
  )
}

export default Background