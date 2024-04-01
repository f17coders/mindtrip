import axios, { AxiosResponse } from 'axios';

// 채팅방 조회
const getPersonalChat = async (
    accessToken: string,
    // memberId : number,
    channelId: string
    ): Promise<any> => {
    try {
        const res: AxiosResponse = await axios.get(`https://mindtrip.site/api/channels/v1/${channelId}`,{
            headers: {
                // "x-member-id": memberId,
                Authorization: `Bearer ${accessToken}`,
            }
        });
        return res.data.result;
    } catch (err) {
        console.log(err);
        throw err;
    }
}



// 채팅방 얻기, 없으면 생성
const registPersonalChat = async (
    accessToken: string,
    consultId: number,
    // receiver: number,
    // memberId: number
    ): Promise<any> => {
    try {
        const res: AxiosResponse = await axios.post(`https://mindtrip.site/api/channels/v1/enter/${consultId}`,
            // "receiver": receiver,
            null
        , {
            headers: {
                // "x-member-id": memberId,
                Authorization: `Bearer ${accessToken}`,
            }
        });
        return res.data.result;
    } catch (err) {
        console.log(err);
        throw err;
    }
}

const send = (stompClient: any, sender: String, message: string, channelId: string): void => {
    try{ 
        if (stompClient && stompClient.connected) {
        const msg = {
            sender: sender,
            text: message,
            channelId: channelId
        };
        stompClient.publish({
            destination: "/pub/api/chat/send/" + channelId,
            body: JSON.stringify(msg)
        });
    }
    } catch(err) {
        console.log(err);
        alert("서버 오류로 인해 메시지 전송이 실패하였습니다!");
    }
};

const entranceChannel = (
    accessToken: string, 
    stompClient: any, message: any, channelId: string): void => {
    if (stompClient && stompClient.connected) {

        stompClient.publish({
            destination: "/pub/api/chat/enter/" + channelId,
            headers: {
                token: accessToken
            },
            body: JSON.stringify(message)
        });
    } else {
        alert("서버 오류로 인해 메시지 전송이 실패하였습니다!");
    }
};

export {
    registPersonalChat,
    getPersonalChat,
    send,
    entranceChannel
}
