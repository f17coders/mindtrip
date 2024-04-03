import { useState } from 'react';
import Modal, { Styles } from 'react-modal';

const customModalStyles: Styles = {
  overlay: {
    backgroundColor: "rgba(0, 0, 0, 0.4)",
    width: "100%",
    height: "100vh",
    zIndex: 10,
    position: "fixed",
    top: 0,
    left: 0,
  },
  content: {
    width: "360px",
    height: "auto",
    zIndex: 150,
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    borderRadius: "10px",
    boxShadow: "2px 2px 2px rgba(0, 0, 0, 0.25)",
    backgroundColor: "white",
    justifyContent: "center",
    overflow: "auto",
  },
};

Modal.setAppElement('#root');

const PostitModal = ({ isOpen, onClose, onSubmit }: { isOpen: boolean, onClose: () => void, onSubmit: (content: string) => void }) => {
  const [content, setContent] = useState('');
  const [inputCount, setInputCount] = useState(0);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const inputValue = e.target.value;
    setContent(inputValue);
    setInputCount(inputValue.length);
  };

  const handleSubmit = () => {
    onSubmit(content);
    setContent('');
    onClose(); // 모달 닫기
  };

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onClose}
      style={customModalStyles}
      contentLabel="Postit Modal"
    >
      <textarea 
        value={content} 
        maxLength={50} 
        onChange={handleInputChange}
        placeholder="여기에 내용을 입력하세요..." 
        style={{ width: "100%", minHeight: "100px", marginBottom: "10px" }} 
      />
      <div style={{ textAlign: 'right', marginRight: '10px', color: '#888', fontSize: '14px' }}>
        {inputCount}/50
      </div>
      <button 
        onClick={handleSubmit} 
        disabled={inputCount === 0 || inputCount > 50} // 입력된 문자 수가 0이거나 50을 초과하면 비활성화
        style={{ 
          width: "100%", 
          padding: "10px", 
          border: "1px solid #0f0f0f", 
          borderRadius: "5px", 
          cursor: "pointer", 
          backgroundColor: (inputCount === 0 || inputCount > 50) ? '#ccc' : '#fff' // 입력된 문자 수에 따라 배경색 변경
        }}
      >
        Submit
      </button>
    </Modal>
  );
};

export default PostitModal;
