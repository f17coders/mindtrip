package com.a303.consultms.domain.channel.service;

import com.a303.consultms.domain.channel.Channel;
import com.a303.consultms.domain.channel.dto.request.ChannelReq;
import com.a303.consultms.domain.channel.dto.response.ChannelListRes;
import com.a303.consultms.domain.channel.dto.response.ChannelRes;
import com.a303.consultms.global.exception.BaseExceptionHandler;
import java.util.List;

public interface ChannelService {

//    String enterConsultingRoom(int consultId, int memberId) throws BaseExceptionHandler;

    Channel readPersonalChatByRecevier(int receiver, int memberId);

    //개인 메시지 조회
//    String registerPersonalChat(int receiver, int sender);

    //참여중인 채팅 목록 조회
    List<ChannelRes> getPersonalChatList(int memberId);

    Channel readPersonalChat(String channelId, int memberId);

    ChannelListRes getSharedChat(String channelId);
}
