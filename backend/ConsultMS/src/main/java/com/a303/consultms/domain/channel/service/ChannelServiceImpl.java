package com.a303.consultms.domain.channel.service;

import static com.a303.consultms.global.exception.code.ErrorCode.NOT_FOUND_CHANNEL_EXCEPTION;

import com.a303.consultms.domain.channel.Channel;
import com.a303.consultms.domain.channel.dto.response.ChannelListRes;
import com.a303.consultms.domain.channel.dto.response.ChannelRes;
import com.a303.consultms.domain.channel.repository.ChannelRepository;
import com.a303.consultms.domain.consult.Consult;
import com.a303.consultms.domain.consult.repository.ConsultRepository;
import com.a303.consultms.domain.member.dto.response.MemberBaseRes;
import com.a303.consultms.domain.message.Message;
import com.a303.consultms.global.client.MemberClient;
import com.a303.consultms.global.exception.BaseExceptionHandler;
import com.a303.consultms.global.exception.code.ErrorCode;
import jakarta.transaction.Transactional;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Service;

@Log4j2
@Service
@RequiredArgsConstructor
public class ChannelServiceImpl implements ChannelService {

    private final ChannelRepository channelRepository;
    private final ConsultRepository consultRepository;
    private final MemberClient memberClient;

    //참여중인 채팅방 목록 조회
    @Override
    public List<ChannelRes> getPersonalChatList(int sender) {

        List<Channel> channelList = channelRepository.findListBySenderOrReceiver(
            String.valueOf(sender)
        );

        List<ChannelRes> channelResList = new ArrayList<>();

        for (Channel channel : channelList) {
            int senderId = Integer.parseInt(channel.getSender().get("memberId"));
            int receiverId = Integer.parseInt(channel.getReceiver().get("memberId"));

            int other = (sender == senderId) ? receiverId : senderId;

            MemberBaseRes receiverInfo = memberClient.getMember(receiverId).getResult();
            MemberBaseRes senderInfo = memberClient.getMember(senderId).getResult();

            //TODO: 탈퇴한 사용자 처리

            ChannelRes channelRes = ChannelRes.builder()
                .channelId(channel.getChannelId())
                .receiver(receiverInfo)
                .sender(senderInfo)
                .build();

            channelResList.add(channelRes);

        }

        return channelResList;
    }

    //개인 채팅 조회
    @Override
    public Channel readPersonalChat(String channelId, int memberId) {

        Channel channel = channelRepository.findById(channelId).get();
        System.out.println(channel);

        Map<String, String> sender = channel.getSender();
        System.out.println(sender);

        if (memberId != Integer.parseInt(sender.get("memberId"))) {
            channel.setSender(channel.getReceiver());
            channel.setReceiver(sender);
        }

//        int receiverId = Integer.parseInt(channel.getReceiver().get("memberId"));

        // TODO: 탈퇴한 사용자 처리

        return channel;
    }

    //채팅방 입장 전 고민상담소 정보 조회
    @Override
    public Channel readPersonalChatByRecevier(int receiver, int memberId) {
        Channel channel = channelRepository.findBySenderOrReceiver(String.valueOf(receiver),
            String.valueOf(memberId));

        return channel;
    }

    //공유된 채팅내역 조회
    @Override
    public ChannelListRes getSharedChat(String channelId)
        throws BaseExceptionHandler {

        //고민 주최자가 누구인지 파악하기
        Consult consult = consultRepository.findByChannelId(channelId);
        int memberId = consult.getMemberId();

        //채널에 대한 정보 가져오기
        Channel channel = channelRepository.findById(channelId)
            .orElseThrow(() -> new BaseExceptionHandler(NOT_FOUND_CHANNEL_EXCEPTION));

        if (channel != null && channel.isShared()) { // 채널이 존재하고 공유된 경우에만 처리
            // 채널의 메시지 리스트를 가져와서 ChannelListRes 객체를 생성하여 반환
            return new ChannelListRes(channel.getChannelId(), memberId, channel.getMessageList());
        } else {
            // 채널이 없거나 공유되지 않은 경우 null 반환
            return null;
        }
    }

}
