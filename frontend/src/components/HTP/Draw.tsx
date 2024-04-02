import { useRef, useState, useEffect, useCallback } from 'react'
import { Button } from '@nextui-org/react';
import axios from 'axios'
import Swal from 'sweetalert2';


type propsType = {
  now: string
  goSurvey: () => void,
  tempAuthorization:string
}


function Draw({now, goSurvey, tempAuthorization}:propsType) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState<boolean>(false)
  const [penColor, setPencolor] = useState<string>('black')

// 좌표 정보를 얻는 함수
  const getCoordinates = (canvas :HTMLCanvasElement, event :MouseEvent|TouchEvent) => {
    if ('touches' in event) {
      // 터치 이벤트의 경우
      const rect = canvas.getBoundingClientRect();
      return {
        offsetX: event.touches[0].clientX - rect.left,
        offsetY: event.touches[0].clientY - rect.top,
      };
    } else {
      // 마우스 이벤트의 경우
      const mouseEvent = event as MouseEvent;
      return {
        offsetX: mouseEvent.offsetX,
        offsetY: mouseEvent.offsetY,
      };
    }
  };

  // 그림 그리기 시작
  const startDrawing = useCallback((e: MouseEvent | TouchEvent): void => {
    // 터치할 때 화면스크롤 안되도록
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.beginPath();
    const { offsetX, offsetY } = getCoordinates(canvas, e);
    ctx.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  }, []);


  // 그려줌
  const draw = useCallback((e: MouseEvent | TouchEvent) => {
    // 터치할 때 화면 스크롤 방지
    e.preventDefault()

    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { offsetX, offsetY } = getCoordinates(canvas, e);
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
    ctx.strokeStyle = penColor
  }, [isDrawing]);

  // 그만 그리면 멈춰주는 함수
  const stopDrawing = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.closePath();
    }
    setIsDrawing(false);
  }, []);

  // 그림 초기화 함수
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas?.height)
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (canvas.width !== canvas.offsetWidth || canvas.height !== canvas.offsetHeight) {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    }
    // 이벤트 리스너 추가 및 제거 함수에서 타입 캐스팅
    const handleEvent = (event: string, listener: (e: MouseEvent) => void, method: 'add' | 'remove') => {
      canvas[`${method}EventListener`](event, listener as EventListener);
    };

    // 마우스로 그릴 때, 이벤트 리스너 추가
    handleEvent('mousedown', startDrawing, 'add');
    handleEvent('mousemove', draw, 'add');
    handleEvent('mouseup', stopDrawing, 'add');
    handleEvent('mouseout', stopDrawing, 'add');

    // 터치 이벤트 리스너 추가
    handleEvent('touchstart', startDrawing, 'add');
    handleEvent('touchmove', draw, 'add');
    handleEvent('touchend', stopDrawing, 'add');
    handleEvent('touchcancel', stopDrawing, 'add');

    return () => {
      // 이벤트 리스너 제거
      handleEvent('mousedown', startDrawing, 'remove');
      handleEvent('mousemove', draw, 'remove');
      handleEvent('mouseup', stopDrawing, 'remove');
      handleEvent('mouseout', stopDrawing, 'remove');
      
      // 마우스 이벤트 리스너 제거
      handleEvent('touchstart', startDrawing, 'remove');
      handleEvent('touchmove', draw, 'remove');
      handleEvent('touchend', stopDrawing, 'remove');
      handleEvent('touchcancel', stopDrawing, 'remove');
    };
  }, [startDrawing, draw, stopDrawing]);


  // 버튼 눌렀을 때 실행되는 함수
  const handleButton = () => {
    const canvas = canvasRef.current
    canvas?.toBlob(function(blob) {
      if (blob){
        const file = new File([blob], `${now}.png`, {type:'image/png'})
        const formData = new FormData()
        formData.append('file', file)
        console.log(file)
        try {
          axios.post(`https://mindtrip.site/api/htp/v1/test/${now}`, formData, {
            headers: {
              Authorization: tempAuthorization
            }
          })
          Swal.fire({
            text: '그림 업로드 완료!',
            icon: "success"
          }).then(() => { goSurvey() })
        } catch(err) {
          console.log(err)
        }
      }
    })
  }


  return (
    <div className="relative w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full cursor-crosshair" />
      <Button 
        onClick={clearCanvas} 
        className='absolute top-3 left-3'
        size='sm'
        variant='light'
        color='secondary'
        endContent={
          <TrashIcon/>
        }
      >초기화</Button>
      <div className='absolute top-3 left-[130px] flex'>
        <div 
          onClick={() => setPencolor('black')}
          className='w-8 h-8 rounded-full bg-black mx-2 hover:cursor-pointer'/>
        <div 
        onClick={() => setPencolor('red')}
        className='w-8 h-8 rounded-full bg-red-500 mx-2 hover:cursor-pointer'/>
        <div 
        onClick={() => setPencolor('#3b82f6')}
        className='w-8 h-8 rounded-full bg-blue-500 mx-2 hover:cursor-pointer'/>
        <div 
        onClick={() => setPencolor('rgb(132 204 22)')}
        className='w-8 h-8 rounded-full bg-lime-400 mx-2 hover:cursor-pointer'/>
        <div 
        onClick={() => setPencolor('rgb(113 63 18)')}
        className='w-8 h-8 rounded-full bg-yellow-900 mx-2 hover:cursor-pointer'/>
      </div>
      <Button
        className={`absolute bottom-0 right-0 m-3 opacity-50`}
        onClick={handleButton}
      >다 그렸어요</Button>
    </div>
  )
}

export default Draw

function TrashIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
      <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
    </svg>
  )
}